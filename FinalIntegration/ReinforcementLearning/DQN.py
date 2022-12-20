import torch


_mov_average_size = 10  # moving average of last 10 epsiodes
ENABLE_WANDB = False
#matplotlib.use("Tkagg")

class DQN(torch.nn.Module):

    def __init__(self, config) -> None:
        super().__init__()
        self.device = torch.device(config.get('device', 'cuda') if torch.cuda.is_available() else 'cpu')
        self.num_inputs = config.get('num_inputs', 3)
        self._hidden = config.get('hidden', [128, 128])
        self.num_actions = config.get('num_actions', 7)

        self.frame_idx = 0
        self.prev_action = None

        modules = []
        modules.append(torch.nn.Linear(self.num_inputs, self._hidden[0]))
        modules.append(torch.nn.ReLU())
        for i in range(0, len(self._hidden) - 1):
            modules.append(torch.nn.Linear(self._hidden[i], self._hidden[i + 1]))
            modules.append(torch.nn.ReLU())
        modules.append(torch.nn.Linear(self._hidden[len(self._hidden) - 1], self.num_actions))

        self.model = torch.nn.Sequential(*modules)

    def forward(self, x):
        return self.model(x)

    def act(self, state, epsilon=0.0):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
            q_value = self.forward(state)
        action = q_value.max(1)[1].data[0].cpu().numpy().tolist()  # argmax over actions
        self.prev_action = action
        return action

