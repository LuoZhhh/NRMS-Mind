import torch
import numpy as np
import gc

from utils import infonce_loss, history_process, candidate_process
from tqdm import tqdm


def train(news, tokenizer, model_user, model_news, device, dataloader, args):
    # ========== setup optim_embeds ========== #
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model_user.parameters()), lr=args.lr, weight_decay=args.weight_decay, eps=1e-8)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epoch)
    for epoch in range(args.num_epoch):
        loss_list = [] 
        print('Epoch:', epoch + 1, 'Training...')
        for history, candidate in tqdm(dataloader):
            # ========== setup optimizer and scheduler ========== #
            loss = loss_compute(news, history, candidate, args.batch_size, model_user, model_news, tokenizer, device, args)
            loss_list.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            del loss; gc.collect(); torch.cuda.empty_cache()
        print('Epoch {:d} finish, loss: {:.4f}'.format(epoch + 1, np.array(loss_list).mean()))
        torch.save(model_user, 'user.pth')
        torch.save(model_news, 'news.pth')



def loss_compute(news, history, candidate, batch, model_user, model_news, tokenizer, device, args):
    loss_all = 0
    for user_index in range(batch):
        # Note that, some users' history is very long, which needs to be cut for efficiency
        # import pdb; pdb.set_trace()
        history_content = history_process(history[user_index], news, args)  # transform the news ID into words
        # import pdb; pdb.set_trace()
        user_tokens = tokenizer(history_content, return_tensors="pt", padding=True)['input_ids'].to(device)  # [len_history, max_news_length]
        # transform the user history into user representation
        h_user = model_user(user_tokens)  # (1, hid_emb)

        # the candidate contains both 1 positive news and 4 negative news
        candidate_content = candidate_process(candidate[user_index], news, args)  # transform the news ID into words
        if candidate_content is None:
            continue
        news_token = tokenizer(candidate_content, return_tensors="pt", padding=True)['input_ids'].to(device)  # news_token -- list (5, max_news_length)
        h_news = model_news(news_token)  # (5, hid_emb)

        loss_all += infonce_loss(h_user, h_news)
        
    
    return loss_all / batch



    






