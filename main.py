from enas.pso import PSO
from enas.utils.data_loader import DataLoader

dl = DataLoader(
    image_size=224,
    batch_size=8,
    channel_num=1,
    dataset=["dataset/train", "dataset/valid"],
)

pso = PSO(n_pop=5, c1=1.2, c2=2.4, w=1.0, exp="exp-6-layers", replace=False)

pso.set_data_loader(data_loader=dl)

pso.init_neural(
    epochs_train=5,
    input_filters=64,
    input_kernel_size=(3, 3),
    input_use_activation=True,
    batch_normalization_rate=0.25,
)

pso.init_search(
    num_cells=3,
    num_nodes=2,
    include_fc=True,
)

pso.search(epochs=30)
