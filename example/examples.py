import numpy as np
import torch
import torch.nn as nn


class DEMNet(nn.Module):
    def __init__(self, response_function,
                 delta_temperature,
                 num_wave=6, num_tbin=43, num_hidden=128):
        super().__init__()

        self.register_buffer(
            "response_function",
            torch.as_tensor(response_function, dtype=torch.float32)
        )

        self.register_buffer(
            "delta_temperature",
            torch.as_tensor(delta_temperature, dtype=torch.float32)
        )

        model = []
        model += [nn.Linear(num_wave,  num_hidden)]
        model += [nn.ReLU()]
        model += [nn.Linear(num_hidden, num_tbin)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        dem = self.model(x)
        recon = dem * self.delta_temperature
        recon = recon @ self.response_function.T
        return dem, recon


if __name__ == "__main__" :

    BATCH_SIZE = 64
    NUM_WAVE = 6
    NUM_TBIN = 43
    NUM_HIDDEN = 128

    delta_temperature = np.random.normal(0, 1, (NUM_TBIN))
    response_function = np.random.normal(0, 1, (NUM_WAVE, NUM_TBIN))

    # print(delta_temperature)
    print(delta_temperature.shape)
    # print(response_function)
    print(response_function.shape)

    my_euv = np.random.normal(0, 1, (BATCH_SIZE, NUM_WAVE))
    print(my_euv.shape)
    my_dem = np.random.normal(0, 1, (BATCH_SIZE, NUM_TBIN))
    print(my_dem.shape)

    recon = my_dem * delta_temperature
    print(recon.shape)
    recon = recon @ response_function.T
    print(recon.shape)

    # model = DEMNet()