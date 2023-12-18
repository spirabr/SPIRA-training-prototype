from spira.core.domain.optimizer import build_optimizer
from spira.core.domain.scheduler import create_scheduler
from spira.tasks.pipeline import config


def train(
    args,
    log_dir,
    checkpoint_path,
    trainloader,
    testloader,
    tensorboard,
    c,
    model_name,
    ap,
    cuda=True,
    model_params=None,
):
    loss1_weight = c.train_config["loss1_weight"]
    use_mixup = False if "mixup" not in c.model else c.model["mixup"]
    if use_mixup:
        mixup_alpha = 1 if "mixup_alpha" not in c.model else c.model["mixup_alpha"]
        mixup_augmenter = Mixup(mixup_alpha=mixup_alpha)
        print("Enable Mixup with alpha:", mixup_alpha)

    model = return_model(c, model_params)

    #
    # if c.train_config['optimizer'] == 'adam':
    #     optimizer = torch.optim.Adam(model.parameters(),
    #                                  lr=c.train_config['learning_rate'], weight_decay=c.train_config['weight_decay'])
    # elif c.train_config['optimizer'] == 'adamw':
    #     optimizer = torch.optim.AdamW(model.parameters(),
    #                                   lr=c.train_config['learning_rate'], weight_decay=c.train_config['weight_decay'])
    # elif c.train_config['optimizer'] == 'radam':
    #     optimizer = RAdam(model.parameters(), lr=c.train_config['learning_rate'], weight_decay=c.train_config['weight_decay'])
    # else:
    #     raise Exception("The %s  not is a optimizer supported" % c.train['optimizer'])
    optimizer = build_optimizer(
        config.train_config.optimizer,
        model.parameters(),
        config.train_config.learning_rate,
        config.train_config.weight_decay,
    )

    step = 0
    if checkpoint_path is not None:
        print("Continue training from checkpoint: %s" % checkpoint_path)
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            model.load_state_dict(checkpoint["model"])
        except:
            print(" > Partial model initialization.")
            model_dict = model.state_dict()
            model_dict = set_init_dict(model_dict, checkpoint, c)
            model.load_state_dict(model_dict)
            del model_dict
        step = 0
    else:
        print("Starting new training run")
        step = 0

    # if c.train_config['lr_decay']:
    #     scheduler = NoamLR(optimizer,
    #                        warmup_steps=c.train_config['warmup_steps'],
    #                        last_epoch=step - 1)
    # else:
    #     scheduler = None
    scheduler = create_scheduler(config, optimizer)

    # convert model from cuda
    if cuda:
        model = model.cuda()

    # define loss function
    if use_mixup:
        criterion = Clip_BCE()
    else:
        criterion = nn.BCELoss()
    eval_criterion = nn.BCELoss(reduction="sum")

    best_loss = float("inf")

    # early stop definitions
    early_epochs = 0

    model.train()
    for epoch in range(c.train_config["epochs"]):
        for feature, target in trainloader:
            if cuda:
                feature = feature.cuda()
                target = target.cuda()

            if use_mixup:
                batch_len = len(feature)
                if (batch_len % 2) != 0:
                    batch_len -= 1
                    feature = feature[:batch_len]
                    target = target[:batch_len]

                mixup_lambda = torch.FloatTensor(
                    mixup_augmenter.get_lambda(batch_len)
                ).to(feature.device)
                output = model(feature[:batch_len], mixup_lambda)
                target = do_mixup(target, mixup_lambda)
            else:
                output = model(feature)
            # Calculate loss
            if c.dataset["class_balancer_batch"] and not use_mixup:
                idxs = target == c.dataset["control_class"]
                loss_control = criterion(output[idxs], target[idxs])
                idxs = target == c.dataset["patient_class"]
                loss_patient = criterion(output[idxs], target[idxs])
                loss = (loss_control + loss_patient) / 2
            else:
                loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # update lr decay scheme
            if scheduler:
                scheduler.step()
            step += 1

            loss = loss.item()
            if loss > 1e8 or math.isnan(loss):
                print("Loss exploded to %.02f at step %d!" % (loss, step))
                break

            # write loss to tensorboard
            if step % c.train_config["summary_interval"] == 0:
                tensorboard.log_training(loss, step)
                if c.dataset["class_balancer_batch"] and not use_mixup:
                    print(
                        "Write summary at step %d" % step,
                        " Loss: ",
                        loss,
                        "Loss control:",
                        loss_control.item(),
                        "Loss patient:",
                        loss_patient.item(),
                    )
                else:
                    print("Write summary at step %d" % step, " Loss: ", loss)

            # save checkpoint file  and evaluate and save sample to tensorboard
            if step % c.train_config["checkpoint_interval"] == 0:
                save_path = os.path.join(log_dir, "checkpoint_%d.pt" % step)
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "step": step,
                        "config_str": str(c),
                    },
                    save_path,
                )
                print("Saved checkpoint to: %s" % save_path)
                # run validation and save best checkpoint
                val_loss = validation(
                    eval_criterion,
                    ap,
                    model,
                    c,
                    testloader,
                    tensorboard,
                    step,
                    cuda=cuda,
                    loss1_weight=loss1_weight,
                )
                best_loss, _ = save_best_checkpoint(
                    log_dir,
                    model,
                    optimizer,
                    c,
                    step,
                    val_loss,
                    best_loss,
                    early_epochs if c.train_config["early_stop_epochs"] != 0 else None,
                )

        print("=================================================")
        print("Epoch %d End !" % epoch)
        print("=================================================")
        # run validation and save best checkpoint at end epoch
        val_loss = validation(
            eval_criterion,
            ap,
            model,
            c,
            testloader,
            tensorboard,
            step,
            cuda=cuda,
            loss1_weight=loss1_weight,
        )
        best_loss, early_epochs = save_best_checkpoint(
            log_dir,
            model,
            optimizer,
            c,
            step,
            val_loss,
            best_loss,
            early_epochs if c.train_config["early_stop_epochs"] != 0 else None,
        )
        if c.train_config["early_stop_epochs"] != 0:
            if early_epochs is not None:
                if early_epochs >= c.train_config["early_stop_epochs"]:
                    break  # stop train
    return best_loss
