




def prune_loop(model, loss, pruner, dataloader, device, sparsity, scope, epochs, train_mode=False):

    # Set model to train or eval mode
    model.train()
    if not train_mode:
        model.eval()

    # Prune model
    for epoch in range(epochs):
        pruner.score(model, loss, dataloader, device)

        sparse = sparsity**((epoch + 1) / epochs)

        pruner.mask(sparse, scope)



pruner = utils.pruner(args.pruner)(masked_parameters(model))  # args.pruner in [Mag, SynFlow, Taylor1ScorerAbs, Rand, SNIP, GraSP]
sparsity = (1.0 - args.rate)**state

if args.pruner == 'synflow':
    iteration_number = 100
else:
    iteration_number = 1

prune_loop(model, criterion, pruner, val_train_loader, torch.device('cuda: {}'.format(args.gpu)),
        sparsity, scope='global', epochs=iteration_number, train_mode=True)

print('sparsity = {}'.format(sparsity))
check_sparsity(model) 
current_mask = extract_mask(model)
torch.save(current_mask, os.path.join(args.save_dir, '{}-mask.pt'.format(state)))
model.load_state_dict(initalization)