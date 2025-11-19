import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import argparse
from tqdm import tqdm
from data.data import *
from torchvision import transforms
from torch.utils.data import DataLoader
from loss.losses import *
from net.ICLR import ICLR

eval_parser = argparse.ArgumentParser(description='Eval')
eval_parser.add_argument('--lol_v1', action='store_true', help='output lolv1 dataset')
eval_parser.add_argument('--lol_v2_real', action='store_true', help='output lol_v2_real dataset')
eval_parser.add_argument('--lol_v2_syn', action='store_true', help='output lol_v2_syn dataset')

eval_parser.add_argument('--alpha', type=float, default=1.0)


ep = eval_parser.parse_args()


def eval(model, testing_data_loader, model_path, output_folder,norm_size=True,LOL=False,v2=False,alpha=1.0):
    torch.set_grad_enabled(False)
    model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
    print('Pre-trained model is loaded.')
    model.eval()
    print('Evaluation:')
    if LOL:
        model.trans.gated = True
    elif v2:
        model.trans.gated2 = True
        model.trans.alpha = alpha
    for batch in tqdm(testing_data_loader):
        with torch.no_grad():
            if norm_size:
                input, name = batch[0], batch[1]
            else:
                input, name, h, w = batch[0], batch[1], batch[2], batch[3]
            
            input = input.cuda()
            output = model(input) 
            
        if not os.path.exists(output_folder):          
            os.mkdir(output_folder)  
            
        output = torch.clamp(output.cuda(),0,1).cuda()
        if not norm_size:
            output = output[:, :, :h, :w]
        
        output_img = transforms.ToPILImage()(output.squeeze(0))
        output_img.save(output_folder + name[0])
        torch.cuda.empty_cache()
    print('===> End evaluation')
    if LOL:
        model.trans.gated = False
    elif v2:
        model.trans.gated2 = False
    torch.set_grad_enabled(True)
    
if __name__ == '__main__':
    
    cuda = True
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, or need to change CUDA_VISIBLE_DEVICES number")
    
    if not os.path.exists('./output'):          
            os.mkdir('./output')  
    
    norm_size = True
    num_workers = 1
    alpha = None
    if ep.lol_v1:
        eval_data = DataLoader(dataset=get_eval_set("./datasets/LOLdataset/eval15/low"), num_workers=num_workers, batch_size=1, shuffle=False)
        output_folder = './output/LOLv1/'
        weight_path = './weights/best.pth'
        
    elif ep.lol_v2_real:
        eval_data = DataLoader(dataset=get_eval_set("./datasets/LOLv2/Real_captured/Test/Low"), num_workers=num_workers, batch_size=1, shuffle=False)
        output_folder = './output/LOLv2_real/'
        weight_path = './weights/best.pth'
        alpha = 0.84
        
    elif ep.lol_v2_syn:
        eval_data = DataLoader(dataset=get_eval_set("./datasets/LOLv2/Synthetic/Test/Low"), num_workers=num_workers, batch_size=1, shuffle=False)
        output_folder = './output/LOLv2_syn/'
        weight_path = './weights/best.pth'
        alpha = ep.alpha
        
    eval_net = ICLR().cuda()
    eval(eval_net, eval_data, weight_path, output_folder, norm_size=norm_size,LOL=ep.lol,v2=ep.lol_v2_real,alpha=alpha)

