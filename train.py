import os
import yaml
import time
import argparse
import importlib
import os.path as osp
from utils import AverageMeter, dict2namespace, update_cfg_hparam_lst
from torch.backends import cudnn
# from torch.utils.tensorboard import SummaryWriter
from utils import SummaryWriter


def get_args():
    # command line args
    parser = argparse.ArgumentParser(
        description='Flow-based Point Cloud Generation Experiment')
    parser.add_argument('config', type=str,
                        help='The configuration file.')

    # distributed training
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use. None means using all '
                             'available GPUs.')

    # Resume:
    parser.add_argument('--resume', default=False, action='store_true')
    parser.add_argument('--pretrained', default=None, type=str,
                        help="Pretrained cehckpoint")
    # Resume if there is a lastest.pt, otherwise don't fail
    parser.add_argument('--soft_resume', default=False, action='store_true')

    # Test run:
    parser.add_argument('--test_run', default=False, action='store_true')
    parser.add_argument('--no_run_time_postfix', default=False, action='store_true')

    # Hyper parameters
    parser.add_argument('--hparams', default=[], nargs="+")
    args = parser.parse_args()

    # parse config file
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config = dict2namespace(config)
    config, hparam_str = update_cfg_hparam_lst(config, args.hparams, strict=False)

    # Currently save dir and log_dir are the same
    if not hasattr(config, "log_dir"):
        #  Create log_name
        if args.test_run:
            cfg_file_name = "test"
        else:
            cfg_file_name = os.path.splitext(os.path.basename(args.config))[0]

        if not args.no_run_time_postfix:
            run_time = time.strftime('%Y-%b-%d-%H-%M-%S')
        else:
            run_time = ""
        post_fix = hparam_str + run_time


        config.log_dir = "logs/%s_%s" % (cfg_file_name, post_fix)
        config.log_name = "logs/%s_%s" % (cfg_file_name, post_fix)
        config.log_name_small = "logs_small/%s_%s" % (cfg_file_name, post_fix)
        config.save_dir = "logs/%s_%s" % (cfg_file_name, post_fix)

        os.makedirs(osp.join(config.log_dir, 'config'), exist_ok=True)
        with open(osp.join(config.log_dir, "config", "config.yaml"), "w") as outf:
            yaml.dump(config, outf)
    return args, config


def main_worker(cfg, args):
    # basic setup
    cudnn.benchmark = True

    # Customized summary writer that write another copy of scalars
    # into a small log_dir (so that it's easier to load for tensorboard)
    writer = SummaryWriter(
        log_dir=cfg.log_name,
        small_log_dir=getattr(cfg, "log_name_small", None))
    data_lib = importlib.import_module(cfg.data.type)
    loaders = data_lib.get_data_loaders(cfg.data, args)
    train_loader = loaders['train_loader']
    test_loader = loaders['test_loader']
    trainer_lib = importlib.import_module(cfg.trainer.type)
    trainer = trainer_lib.Trainer(cfg, args)

    start_epoch = 0
    if args.resume or args.soft_resume:
        if args.pretrained is not None:
            start_epoch = trainer.resume(args.pretrained)
        else:
            latest = osp.join(cfg.log_dir, "latest.pt")
            if osp.isfile(latest) or not args.soft_resume:
                # If the file doesn't exist, and soft resume is not specified
                # then it will throw errors.
                start_epoch = trainer.resume(latest)

    # If test run, go through the validation loop first
    if args.test_run:
        trainer.save(epoch=-1, step=-1)
        val_info = trainer.validate(test_loader, epoch=-1)
        trainer.log_val(val_info, writer=writer, epoch=-1)

    # main training loop
    print("Start epoch: %d End epoch: %d" % (start_epoch, cfg.trainer.epochs))
    step = 0
    duration_meter = AverageMeter("Duration")
    updatetime_meter = AverageMeter("Update")
    loader_meter = AverageMeter("Loader time")
    logtime_meter = AverageMeter("Log time")
    for epoch in range(start_epoch, cfg.trainer.epochs):

        # train for one epoch
        iter_start = time.time()
        loader_start = time.time()
        for bidx, data in enumerate(train_loader):
            loader_duration = time.time() - loader_start
            loader_meter.update(loader_duration)

            start_time = time.time()
            step = bidx + len(train_loader) * epoch + 1
            logs_info = trainer.update(data)
            duration = time.time() - start_time
            updatetime_meter.update(duration)

            logtime_start = time.time()
            if step % int(cfg.viz.log_freq) == 0 and int(cfg.viz.log_freq) > 0:
                print("Epoch %d Batch [%2d/%2d] Time/Iter: Train[%3.2fs] "
                      "Update[%3.2fs] Log[%3.2fs] Load[%3.2fs] Loss %2.5f"
                      % (epoch, bidx, len(train_loader),
                         duration_meter.avg,
                         updatetime_meter.avg, logtime_meter.avg,
                         loader_meter.avg, logs_info['loss']))
                visualize = step % int(cfg.viz.viz_freq) == 0 and \
                            int(cfg.viz.viz_freq) > 0
                trainer.log_train(
                    logs_info, data,
                    writer=writer, epoch=epoch, step=step, visualize=visualize)
            logtime_duration = time.time() - logtime_start
            logtime_meter.update(logtime_duration)
            iter_duration = time.time() - iter_start
            duration_meter.update(iter_duration)

            # Reset loader time
            loader_start = time.time()

        # Save first so that even if the visualization bugged,
        # we still have something
        if (epoch + 1) % int(cfg.viz.save_freq) == 0 and \
                int(cfg.viz.save_freq) > 0:
            trainer.save(epoch=epoch, step=step)

        if (epoch + 1) % int(cfg.viz.val_freq) == 0 and \
                int(cfg.viz.val_freq) > 0:
            val_info = trainer.validate(test_loader, epoch=epoch)
            trainer.log_val(val_info, writer=writer, epoch=epoch)

        # Signal the trainer to cleanup now that an epoch has ended
        trainer.epoch_end(epoch, writer=writer)

    # Final round of validation
    val_info = trainer.validate(test_loader, epoch=epoch + 1)
    trainer.log_val(val_info, writer=writer, epoch=epoch + 1)
    trainer.save(epoch=epoch, step=step)
    writer.close()


if __name__ == '__main__':
    # command line args
    args, cfg = get_args()

    print("Arguments:")
    print(args)

    print("Configuration:")
    print(cfg)

    main_worker(cfg, args)
