import torch

def train_model(
    model,
    optimizer,
    criterion,
    train_loader,
    val_loader,
    test_loader,
    epochs,
    device,
    output_dir,
    log_file,
    print_freq,
):
    train_loss_history = []
    val_loss_history = []
    for epoch in range(epochs):
        dashes = '-' * 20
        print(f'Epoch {epoch + 1}/{epochs} {dashes}')
        with open(log_file, 'a') as writer:
            writer.write(f'Epoch {epoch + 1}/{epochs} {dashes}\n')
        # Training loop
        model.train()
        running_loss_avg = 0
        running_count = 0
        for i, (sample_a, sample_b) in enumerate(train_loader):
            optimizer.zero_grad()
            sample_a = sample_a.to(device)
            sample_b = sample_b.to(device)
            za = model(sample_a)
            zb = model(sample_b)
            loss = criterion(za, zb)
            # print('train', loss.item())
            n = running_count
            m = n + sample_a.shape[0]
            running_loss_avg = ((n * running_loss_avg) + loss.item()) / m
            running_count = m
            # running_loss_avg = (
            #     (running_loss_avg * running_count) + (loss.item() * sample_a.shape[0])
            # ) / (running_count + sample_a.shape[0])
            # running_count += sample_a.shape[0]
            if (i + 1) % print_freq == 0:
                print(
                    'Training |',
                    f'Epoch: {epoch + 1}/{epochs} |',
                    f'Step: {i + 1}/{len(train_loader)} |',
                    f'Loss: {running_loss_avg}'
                )
                with open(log_file, 'a') as writer:
                    writer.writelines(
                        [
                            'Training | ',
                            f'Epoch: {epoch + 1}/{epochs} | ',
                            f'Step: {i + 1}/{len(train_loader)} | ',
                            f'Loss: {running_loss_avg}\n'
                        ]
                    )
                train_loss_history.append(
                    (
                        epoch * len(train_loader) + (i + 1),
                        running_loss_avg
                    )
                )
            loss.backward()
            optimizer.step()

        # Validation loop
        model.eval()
        val_running_loss_avg = 0
        val_running_count = 0
        with torch.no_grad():
            for sample_a, sample_b in val_loader:
                sample_a = sample_a.to(device)
                sample_b = sample_b.to(device)
                za = model(sample_a)
                zb = model(sample_b)
                loss = criterion(za, zb)
                # print('val', loss.item())
                n = val_running_count
                m = n + sample_a.shape[0]
                val_running_loss_avg = ((n * val_running_loss_avg) + loss.item()) / m
                val_running_count = m
                # val_running_loss = ((val_running_loss * val_running_count) + (loss.item() * sample_a.shape[0])) / (val_running_count + sample_a.shape[0])
                # val_running_count += sample_a.shape[0]
        print(
            'Validation |',
            f'Epoch: {epoch + 1} |',
            f'Loss: {val_running_loss_avg}'
        )
        with open(log_file, 'a') as writer:
            writer.writelines(
                [
                    'Validation | ',
                    f'Epoch: {epoch + 1} | ',
                    f'Loss: {val_running_loss_avg}\n'
                ]
            )
        val_loss_history.append(
            (
                (epoch + 1) * len(train_loader),
                val_running_loss_avg
            )
        )

    # Test loop
    dashes = '-' * 20
    print(dashes)
    with open(log_file, 'a') as writer:
        writer.write(f'{dashes}\n')
    model.eval()
    test_loss_list = []
    test_running_loss_avg = 0
    test_running_count = 0
    with torch.no_grad():
        for sample_a, sample_b in test_loader:
            sample_a = sample_a.to(device)
            sample_b = sample_b.to(device)
            za = model(sample_a)
            zb = model(sample_b)
            loss = criterion(za, zb)
            # print('test', loss.item())
            n = test_running_count
            m = n + sample_a.shape[0]
            test_running_loss_avg = ((n * test_running_loss_avg) + loss.item()) / m
            test_running_count = m
            # test_loss_list.append(loss.item() / sample_a.shape[0])
    print(
        'Testing |',
        f'Loss: {test_running_loss_avg}'
    )
    with open(log_file, 'a') as writer:
        writer.writelines(
            [
                'Testing | ',
                # f'Loss: {sum(test_loss_list) / len(test_loss_list)}\n'
                f'Loss: {test_running_loss_avg}\n'
            ]
        )
    
    return model, train_loss_history, val_loss_history, test_running_loss_avg