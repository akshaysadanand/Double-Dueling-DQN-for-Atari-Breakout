def add_arguments(parser):
    '''
    Add your arguments here if needed. The TA will run test.py to load
    your default arguments.

    For example:
        parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
        parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate for training')
    '''
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate for training')
    parser.add_argument('--gamma', type=float, default=0.99)
    #parser.add_argument('--n_actions', type=int, default=4, help='Number of Actions')
    #parser.add_argument('--input_dims', type=int, default=84, help='Number of Actions')
    parser.add_argument('--algo', type=str, default='DuelingDDQNAgent')
    parser.add_argument('--mem_size', type=int, default=100000)
    #parser.add_argument('--env_name', type=str, default='BreakoutNoFrameskip-v4')
    parser.add_argument('--eps_min', type=float, default=0.01)
    parser.add_argument('--eps_dec', type=float, default=1e-6)
    parser.add_argument('--replace', type=int, default=10000)
    parser.add_argument('--chkpt_dir', type=str, default='models/')
    parser.add_argument('--eps', type=float, default=1.0,help='1')
    parser.add_argument('--load_checkpoint', type=bool, default=False, help='load trained model')
    
    return parser
