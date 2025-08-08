import monai
from monai.data import SmartCacheDataset, CacheDataset

def define_Dataset(opt, return_filepaths=False, apply_split=True):
    dataset_opt = opt['dataset_opt']
    dataset_name = dataset_opt['name']
    dataset_type = dataset_opt['dataset_type']

    dataset_params_train = dataset_opt['train_dataset_params']
    dataset_params_test = dataset_opt['test_dataset_params']

    # -----------------------------------------
    # super-resolution datasets
    # -----------------------------------------
    if dataset_name == "LUND":
        from data.Dataset_LUND import Dataset_LUND as D
        dataset = D(opt)
        train_files, test_files = dataset.get_file_paths()
        transforms = dataset.get_transforms(mode="train")
        test_transforms = dataset.get_transforms(mode="test")
        baseline_transforms = dataset.get_transforms(mode="test", baseline=True)
        data_path = dataset.data_path

    elif dataset_name == "VoDaSuRe_OME":
        from data.Dataset_VoDaSuRe_OME import Dataset_VoDaSuRe_OME as D
        dataset = D(opt)
        dataset_dict_train, dataset_dict_test = dataset.get_dataset_dicts()
        #train_files, test_files = dataset.get_file_paths()
        transforms = dataset.get_transforms(mode="train")
        test_transforms = dataset.get_transforms(mode="test")
        #baseline_transforms = dataset.get_baseline_transforms(mode="test")
        data_path = dataset.data_path

    elif dataset_name == "VoDaSuRe":
        from data.Dataset_VoDaSuRe import Dataset_VoDaSuRe as D
        dataset = D(opt)
        train_files, test_files = dataset.get_file_paths()
        transforms = dataset.get_transforms(mode="train")
        test_transforms = dataset.get_transforms(mode="test")
        baseline_transforms = dataset.get_baseline_transforms(mode="test")
        data_path = dataset.data_path

    elif dataset_name == "FEMur":
        from data.Dataset_FEMur import Dataset_FEMur as D
        dataset = D(opt, apply_split)
        train_files, test_files = dataset.get_file_paths()
        transforms = dataset.get_transforms(mode="train")
        test_transforms = dataset.get_transforms(mode="test")
        baseline_transforms = dataset.get_baseline_transforms(mode="test")
        data_path = dataset.data_path

    elif dataset_name == "Binning_Brain":
        from data.Dataset_Binning_Brain import Dataset_Binning_Brain as D
        dataset = D(opt, apply_split)
        train_files, test_files = dataset.get_file_paths()
        transforms = dataset.get_transforms(mode="train")
        test_transforms = dataset.get_transforms(mode="test")
        baseline_transforms = dataset.get_baseline_transforms(mode="test")
        data_path = dataset.data_path

    elif dataset_name == "Binning_Bone":
        # from data.Dataset_Binning_Bone import Dataset_Binning_Bone as D
        # dataset = D(opt, apply_split)
        from data.Dataset_OME import Dataset_OME as D
        dataset = D(opt)
        train_files, test_files = dataset.get_file_paths()
        transforms = dataset.get_transforms(mode="train")
        test_transforms = dataset.get_transforms(mode="test")
        baseline_transforms = dataset.get_transforms(mode="test", baseline=True)
        data_path = dataset.data_path

    elif dataset_name == "Synthetic_2022_QIM_52_Bone":
        from data.Dataset_2022_QIM_52_Bone import Dataset_2022_QIM_52_Bone as D
        dataset = D(opt, apply_split)
        train_files, test_files = dataset.get_file_paths()
        transforms = dataset.get_transforms(mode="train")
        test_transforms = dataset.get_transforms(mode="test")
        baseline_transforms = dataset.get_baseline_transforms(mode="test")
        data_path = dataset.data_path

    elif dataset_name == "2022_QIM_52_Bone":
        from data.Dataset_2022_QIM_52_Bone import Dataset_2022_QIM_52_Bone as D
        dataset = D(opt, apply_split)
        train_files, test_files = dataset.get_file_paths()
        transforms = dataset.get_transforms(mode="train")
        test_transforms = dataset.get_transforms(mode="test")
        baseline_transforms = dataset.get_baseline_transforms(mode="test")
        data_path = dataset.data_path

    elif dataset_name == "HCP_1200":
        from data.Dataset_HCP_1200 import Dataset_HCP_1200 as D
        dataset = D(opt)
        train_files, test_files = dataset.get_file_paths()
        transforms = dataset.get_transforms(mode="train")
        test_transforms = dataset.get_transforms(mode="test")
        baseline_transforms = dataset.get_baseline_transforms(mode="test")
        data_path = dataset.data_path

    elif dataset_name == "HCP_1200_new":
        from data.Dataset_HCP_1200_new import Dataset_HCP_1200 as D
        dataset = D(opt)
        train_files, test_files = dataset.get_file_paths()
        transforms = dataset.get_transforms(mode="train")
        test_transforms = dataset.get_transforms(mode="test")
        baseline_transforms = dataset.get_transforms(mode="test", baseline=True)
        data_path = dataset.data_path

    elif dataset_name == "IXI":
        from data.Dataset_IXI import Dataset_IXI as D
        dataset = D(opt)
        train_files, test_files = dataset.get_file_paths()
        transforms = dataset.get_transforms(mode="train")
        test_transforms = dataset.get_transforms(mode="test")
        baseline_transforms = dataset.get_baseline_transforms(mode="test")
        data_path = dataset.data_path

    elif dataset_name == "BRATS2023":
        from data.Dataset_BRATS2023 import Dataset_BRATS2023 as D
        dataset = D(opt)
        train_files, test_files = dataset.get_file_paths()
        transforms = dataset.get_transforms(mode="train")
        test_transforms = dataset.get_transforms(mode="test")
        baseline_transforms = dataset.get_baseline_transforms(mode="test")
        data_path = dataset.data_path

    elif dataset_name == "KIRBY21":
        from data.Dataset_KIRBY21 import Dataset_KIRBY21 as D
        dataset = D(opt)
        train_files, test_files = dataset.get_file_paths()
        transforms = dataset.get_transforms(mode="train")
        test_transforms = dataset.get_transforms(mode="test")
        baseline_transforms = dataset.get_baseline_transforms(mode="test")
        data_path = dataset.data_path

    elif dataset_name == "CTSpine1K":
        from data.Dataset_CTSpine1K import Dataset_CTSpine1K as D
        dataset = D(opt)
        train_files, test_files = dataset.get_file_paths()
        transforms = dataset.get_transforms(mode="train")
        test_transforms = dataset.get_transforms(mode="test")
        baseline_transforms = dataset.get_baseline_transforms(mode="test")
        data_path = dataset.data_path

    elif dataset_name == "LITS":
        from data.Dataset_LITS import Dataset_LITS as D
        dataset = D(opt)
        train_files, test_files = dataset.get_file_paths()
        transforms = dataset.get_transforms(mode="train")
        test_transforms = dataset.get_transforms(mode="test")
        baseline_transforms = dataset.get_baseline_transforms(mode="test")
        data_path = dataset.data_path

    elif dataset_name == "LIDC_IDRI":
        from data.Dataset_LIDC_IDRI import Dataset_LIDC_IDRI as D
        dataset = D(opt)
        train_files, test_files = dataset.get_file_paths()
        transforms = dataset.get_transforms(mode="train")
        test_transforms = dataset.get_transforms(mode="test")
        baseline_transforms = dataset.get_baseline_transforms(mode="test")
        data_path = dataset.data_path

    else:
        raise NotImplementedError('Dataset %s not implemented.' % dataset_name)

    if dataset_type == "MonaiSmartCacheDataset":
        train_dataset = SmartCacheDataset(train_files,
                                          transform=transforms,
                                          #transform=cst,
                                          replace_rate=dataset_params_train['replace_rate'],
                                          cache_num=dataset_params_train['cache_num'],
                                          cache_rate=1.0,
                                          num_init_workers=dataset_params_train['init_workers'],
                                          num_replace_workers=dataset_params_train['replace_workers'],
                                          copy_cache=False,
                                          shuffle=True
                                          )

        test_dataset = SmartCacheDataset(test_files,
                                         transform=test_transforms,
                                         #transform=cst,
                                         replace_rate=dataset_params_test['replace_rate'],
                                         cache_num=dataset_params_test['cache_num'],
                                         cache_rate=1.0,
                                         num_init_workers=dataset_params_test['init_workers'],
                                         num_replace_workers=dataset_params_test['replace_workers'],
                                         copy_cache=False,
                                         shuffle=False
                                         )

        baseline_dataset = monai.data.Dataset(test_files, transform=baseline_transforms)

    elif dataset_type == "CacheDataset":
        train_dataset = CacheDataset(train_files,
                                      transform=transforms,
                                      # transform=cst,
                                      cache_num=dataset_params_train['cache_num'],
                                      cache_rate=1.0,
                                      num_workers=dataset_params_train['init_workers'],
                                      copy_cache=False,
                                      as_contiguous=True,
                                      )

        test_dataset = CacheDataset(test_files,
                                     transform=test_transforms,
                                     cache_num=dataset_params_test['cache_num'],
                                     cache_rate=1.0,
                                     num_workers=dataset_params_test['init_workers'],
                                     copy_cache=False,
                                     as_contiguous=True,
                                     )

        baseline_dataset = monai.data.Dataset(test_files, transform=baseline_transforms)

    elif dataset_type == "DDP_CacheDataset":
        from data.dataset import DDP_CacheDataset
        train_dataset = DDP_CacheDataset(train_files,
                                      transform=transforms,
                                      cache_rate=1.0,
                                      num_workers=dataset_params_train['init_workers'])

        test_dataset = DDP_CacheDataset(test_files,
                                     transform=test_transforms,
                                     cache_rate=1.0,
                                     num_workers=dataset_params_test['init_workers'])

        baseline_dataset = monai.data.Dataset(test_files, transform=baseline_transforms)

    elif dataset_type == "MonaiDataset":
        train_dataset = monai.data.Dataset(train_files, transform=transforms)
        test_dataset = monai.data.Dataset(test_files, transform=test_transforms)
        baseline_dataset = monai.data.Dataset(test_files, transform=baseline_transforms)

    elif dataset_type == "MonaiIterableDataset":
        train_dataset = monai.data.IterableDataset(train_files, transform=transforms)
        test_dataset = monai.data.IterableDataset(test_files, transform=transforms)
        baseline_dataset = monai.data.Dataset(test_files, transform=baseline_transforms)

    elif dataset_type == "MonaiCacheDataset":
        train_dataset = monai.data.CacheDataset(train_files, transform=transforms, num_workers=1)
        test_dataset = monai.data.CacheDataset(test_files, transform=transforms, num_workers=1)
        baseline_dataset = monai.data.Dataset(test_files, transform=baseline_transforms)

    elif dataset_type == "ZarrDataset":
        from data.ZarrIterableDataset import ZarrIterableDataset
        group_name = dataset.group_name
        ome_levels = ['0', '2']  # Example levels, adjust as needed
        patch_shape = (opt.dataset_opt.patch_size, opt.dataset_opt.patch_size, opt.dataset_opt.patch_size)

        train_dataset = ZarrIterableDataset(ome_levels,
                                    group_name,
                                    paths=train_files,
                                    patch_shape=patch_shape,
                                    patch_transform=transforms,
                                    num_workers=4,
                                    queue_size=256,
                                    store_type='DirectoryStore')

        test_dataset = ZarrIterableDataset(ome_levels,
                                    group_name,
                                    paths=test_files,
                                    patch_shape=patch_shape,
                                    patch_transform=transforms,
                                    num_workers=4,
                                    queue_size=256,
                                    store_type='DirectoryStore',
                                    num_samples=opt.train_opt.validation_iterations)

        baseline_dataset = ZarrIterableDataset(ome_levels,
                                    group_name,
                                    paths=test_files,
                                    patch_shape=patch_shape,
                                    patch_transform=transforms,
                                    num_workers=1,
                                    queue_size=256,
                                    store_type='DirectoryStore',
                                    num_samples=opt.train_opt.validation_iterations)

    elif dataset_type == "ZarrDatasetV2":
        from data.ZarrIterableDatasetV2 import ZarrIterableDataset

        patch_shape = (opt.dataset_opt.patch_size, opt.dataset_opt.patch_size, opt.dataset_opt.patch_size)
        patch_shape_hr = (opt.dataset_opt.patch_size_hr, opt.dataset_opt.patch_size_hr, opt.dataset_opt.patch_size_hr)
        train_dataset = ZarrIterableDataset(dataset_dict_train,
                                            patch_shape,
                                            patch_shape_hr,
                                            patch_transform=transforms,
                                            up_factor=opt.up_factor,
                                            num_workers=8,
                                            queue_size=128,
                                            store_type='DirectoryStore',
                                            num_samples=opt.train_opt.iterations//10,
                                            sampling_method='random'  # 'random' or 'in_chunk'
                                            )

        test_dataset = ZarrIterableDataset(dataset_dict_test,
                                           patch_shape,
                                           patch_shape_hr,
                                           patch_transform=test_transforms,
                                           up_factor=opt.up_factor,
                                           num_workers=8,
                                           queue_size=128,
                                           store_type='DirectoryStore',
                                           num_samples=opt.train_opt.validation_iterations,
                                           sampling_method='random'  # 'random' or 'in_chunk'
                                           )

        baseline_dataset = test_dataset

    else:
        raise NotImplementedError('Dataset type %s is not found.' % dataset_type)

    print('Dataset %s of type %s is created.' % (dataset_name, dataset_type))
    if return_filepaths:
        return train_dataset, test_dataset, baseline_dataset, data_path, train_files, test_files
    else:
        return train_dataset, test_dataset, baseline_dataset
