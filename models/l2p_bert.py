import torch 
import torch.nn as nn 
import transformers
import torch.nn.functional as F
from models.utils.continual_model import ContinualModel
from utils.args import *
from utils.prompt_pool import PromptPool
transformers.logging.set_verbosity(50)

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='l2p_vit')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    parser.add_argument('--alpha', type=float, required=True,
                        help='Penalty weight.')
    parser.add_argument('--beta', type=float, required=True,
                        help='Penalty weight.')
    parser.add_argument('--pw',type=float,default=0.5,help='Penalty weight.')
    parser.add_argument('--freeze_clf',type=int,default=0,help='clf freeze flag')
    parser.add_argument('--init_type',type=str,default='default',help='prompt & key initialization')

    return parser

class L2PBert(ContinualModel):
    NAME = 'l2p_bert'
    COMPATIBILITY = ['class-il']

    def __init__(
        self, 
        backbone, 
        loss, 
        args, 
    ):
        super(L2PBert, self).__init__(backbone, loss, args, None)
        # self.net.config.type_vocab_size = 4
        # token_embed = nn.Embedding(self.net.config.type_vocab_size, self.net.config.hidden_size)
        # token_embed.weight.data.uniform_(-1,1)
        # self.net.bert.embeddings.token_type_embeddings = token_embed

        self.bertEmbeddings = self.net.bert.embeddings
        self.bertEncoder = self.net.bert.encoder
        self.classifier = self.net.classifier 

        self.net.requires_grad_(False)
        self.bertEmbeddings.requires_grad_(False)
        self.bertEncoder.requires_grad_(False)
        self.classifier.requires_grad_(False)

        if args.freeze_clf == 0:
            self.classifier.requires_grad_(True)
        else:
            self.classifier.requires_grad_(False)
        
        self.learning_param = None 
        self.args = args 
        self.lr = args.lr 

        self.topN = 5 
        self.prompt_num = 5 # length of prompt 
        self.pool_size = 10 # prompt size


        self.pool = PromptPool()
        self.pool.initPool(
            layer = 1, 
            total = self.pool_size, 
            pnum = self.prompt_num, 
            pdim = 768, 
            kdim = 768, 
            device = self.device, 
            embedding_layer = None, 
            init_type = args.init_type
        )

        self.init_opt(args)
        
    def init_opt(self, args):
        self.pool.key_freq_past = self.pool.key_freq_now.clone().detach()
        key_list = [e for layer_k in self.pool.key_list for e in layer_k]
        prompt_list = [e for layer_p in self.pool.prompt_list for e in layer_p] 

        if args.freeze_clf == 0:
            self.learning_param = key_list + prompt_list + list(self.classifier.parameters())
            self.opt = torch.optim.AdamW(
                params = self.learning_param, 
                lr = self.lr 
            )
        else:
            self.learning_param = key_list + prompt_list 
            self.opt = torch.optim.AdamW(
                params = self.learning_param, 
                lr = self.lr
            )
    
    def similarity(self, pool, q, k, topN):
        q = nn.functional.normalize(q, dim = -1)
        # batch x 768 
        k = nn.functional.normalize(k, dim = -1)
        # pool_size x 768 = 10x768 = Tx 768 
        sim = torch.matmul(q, k.T)
        # batch x T 
        dist = 1 - sim  
        # batch x T 
        val, idx = torch.topk(dist, topN, dim = 1, largest=False)
        dist_pick = []
        
        for b in range(idx.shape[0]):
            pick = []
            for i in range(idx.shape[1]):
                pick.append(dist[b][idx[b][i]])
            dist_pick.append(torch.stack(pick))    

        dist = torch.stack(dist_pick)
        return dist, idx 
    
    def getPrompts(self, pool, query):
        B, D = query.shape  
        pTensor = torch.stack(pool.prompt_list[0])
        # pool_size x 5 x 768 = 10 x 5 x 768 
        kTensor = torch.stack(pool.key_list[0])
        # pool_size x 768 
        T, Lp, Dp = pTensor.shape
        T, D = kTensor.shape 
        # ! selectedKeys : (B, topN)    
        distance, selectedKeys = self.similarity(pool, query, kTensor, self.topN)
        prompts = pTensor[selectedKeys, :, :]
        # B x topN x Lp Dp = B x 5 x 5 x 768
        prompts = prompts.reshape(B, -1, Dp)
        return prompts, distance, selectedKeys 

    def bertLayer(
        self, 
        input_ids, 
        prompt_length, 
        boundary = None
    ):
        # B, 0:N*Lp, D => POOLING -> B, D 
        z_prompted = self.layernorm(
            self.bertEncoder(input_ids)[0]
        )[:, 1:prompt_length+1, :]
        z_clf = torch.mean(z_prompted, dim = 1)
        # batch x D 
        return self.classifier(z_clf), z_clf 
    
    def forward_l2p(
        self, 
        input_ids: torch.Tensor,
        task_id:int = None
    ):
        input_ids = input_ids.cuda()
        import pickle
        print('input_ids  ', input_ids)
        from transformers import AutoModelForSequenceClassification
        test_model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased')
        print('test model : ', test_model(input_ids))
        print('output : ', self.net(input_ids))
        torch.save(input_ids, 'input_ids.pt')
        token_embedding = self.bertEmbeddings(input_ids)
        print('embedding shape : ', token_embedding.shape)
        # print(self.bertEncoder)
        print('token embedding : ', token_embedding)
        print(token_embedding)
        representations = self.bertEncoder(token_embedding[:, 1:, :])
        print('representations shape: ', representations.shape)
        query = representations[:, 0, :]
        prompts, distance, selectedKeys = self.getPrompts(self.pool, query) 
        B, NLp, Dp = prompts.shape 
        prompted_x = torch.cat(
            [
                token_embedding[:,0, :].unsqeeze(1), 
                prompts,
                token_embedding[:, 1:, :]
            ], 
            1
        )

        # cls + prompt + input
        logits, z_clf = self.bertLayer(
            input_ids = prompted_x,
            prompt_length = NLp
        )

        return logits, distance, z_clf


    def forward_model(
        self, 
        input_ids : torch.Tensor, 
        task_id = None
    ):
        if self.pool == None:
            return self.net(input_ids = input_ids, task_id = task_id)
        logits, distance, z_clf = self.forward_l2p(
            input_ids = input_ids,
            task_id = task_id
        )

        return logits
    
    def observe(
        self, 
        input_ids: torch.Tensor,
        labels,
        dataset, 
        t
    ):
        logits, distance, z_clf = self.forward_l2p(
            input_ids = input_ids, 
        )
        logits_original = logits.clone().detach()
        logits[:, 0:t*dataset.N_CLASSES_PER_TASK] = -float('inf')
        logits[:, (t+1)*dataset.N_CLASSES_PER_TASK:] = -float('inf')
        loss = self.loss(logits, labels) + self.args.pw * torch.mean(torch.sum(distance, dim = 1))

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return loss.item()