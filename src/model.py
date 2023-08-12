

## Here, you should build the neural network. A simple model is given below as an example, you can modify the neural network architecture.

class FoInternNet(nn.Module):
    def __init__(self, input_size, n_classes):
        super(FoInternNet, self).__init__()
        self.input_size = input_size
        self.n_classes = n_classes

        # Here, layers are given as example, please feel free to modify them. 
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=n_classes, kernel_size=1, stride=1)

    def forward(self, x):
        """This function feeds input data into the model layers defined.
        Args:
            x : input data
        """
        #########################################
        # CODE
        #########################################
        return x



if __name__ == '__main__':
    model = FoInternNet(input_size=(HEIGHT, WIDTH), n_classes=N_CLASS)
