# convtran_with_sal_final_ver1
python main.py --mode=test --sal_mode=NJU2K --test_root=../testsod/NJU2K_test/NJU2K_test --test_list=../testsod/NJU2K_test/NJU2K_test/test.lst --test_folder=./test180/NJU2K --model=./checkpoints_whole/demo-07/epoch_140.pth --batch_size=1

python main.py --mode=test --sal_mode=STERE --test_root=../testsod/STERE/STERE --test_list=../testsod/STERE/STERE/test.lst --test_folder=./test_r/STERE --model=./checkpoints_whole/demo-07/epoch_140.pth  --batch_size=1



python main.py --mode=test --sal_mode=SIP --test_root=../testsod/SIP/SIP --test_list=../testsod/SIP/SIP/test.lst --test_folder=./test_wo_edge_loss/SIP --model=./checkpoints/demo-21/epoch_25.pth --batch_size=1



 python main.py --mode=test --model=./checkpoints/demo-07/epoch_180.pth --sal_mode=NLPR --test_root=../testsod/NLPR/NLPR --test_list=../testsod/NLPR/NLPR/test.lst --test_folder=./test180/NLPR --batch_size=1 




python main.py --mode=test --sal_mode=LFSD --test_root=../testsod/LFSD/LFSD --test_list=../testsod/LFSD/LFSD/test.lst --test_folder=./test_r/LFSD --model=./checkpoints_whole/demo-07/epoch_140.pth  --batch_size=1

python main.py --mode=test --sal_mode=RGBD135 --test_root=../testsod/RGBD135/RGBD135 --test_list=../testsod/RGBD135/RGBD135/test.lst --test_folder=./test_wo_edge_loss/RGBD135 --model=./checkpoints/demo-21/epoch_25.pth --batch_size=1



