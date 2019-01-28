from LunarLander import LunarLander
import gym

def generate_exp1_n_2():
    # execute training
    from datetime import datetime
    print('===== Start =====')
    print('start time: ', str(datetime.now()))
    lander = LunarLander(annealing_size=150, alpha=0.0015, batch_size=25, update_step=2, load_weights=False)
    lander.start_record(render=False)
    lander.train()
    lander.test()
    lander.end_record(upload_key='sk_XwbuJNCrQnqa1MJDOk3dyQ')
    print('completed time: ', str(datetime.now()))

def generate_exp3(annealing_size=150, alpha=0.0015, batch_size=25, update_step=2):
    # execute training
    from datetime import datetime
    print('===== Start =====')
    print('start time: ', str(datetime.now()))
    lander = LunarLander(annealing_size=annealing_size, alpha=alpha, batch_size=batch_size, update_step=update_step, load_weights=False)
    lander.start_record(render=False)
    lander.train()
    lander.test()
    lander.end_record(upload_key='sk_XwbuJNCrQnqa1MJDOk3dyQ')
    print('completed time: ', str(datetime.now()))

def generate_exp3_full():
    # executing this function will take a long time....
    for a in range([0.0010, 0.0015, 0.0020]):
        for n in range([150, 250, 500]):
            for b in range([25, 50]):
                for c in range([2, 5]):
                    print('**** running exp 3 at: a = ', a, ' | N = ', n, ' | b = ', b, ' | c = ', c, ' ******')
                    generate_exp3(annealing_size=n, alpha=a, batch_size=b, update_step=c)


if __name__ == '__main__':
    # select experiments to run
    generate_exp1_n_2()
    # generates sub sample of exp3
    # generate_exp3(annealing_size=150, alpha=0.0015, batch_size=25, update_step=2)
    # generate full exp3
    # generate_exp3_full()