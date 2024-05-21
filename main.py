import os
import sys
sys.path.append(os.path.dirname(__file__))

from training import train_gcn, pretrain_F, train_uinj


if __name__ == '__main__':
    # train_gcn.run()
    # train_gcn.test()
    pretrain_F.run()
    train_uinj.run()
    train_uinj.test()

    
    # train_uinj.exp2()
    train_uinj.exp1()
    train_uinj.exp3(800)
    train_uinj.exp3(1200)
    train_uinj.exp3(1500)
    
