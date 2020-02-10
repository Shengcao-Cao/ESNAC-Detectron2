from ESNAC.architecture import comp_action_rand, comp_rep
import ESNAC.options as opt

def random_search(arch, kernel, search_n=opt.ac_search_n):
    action_best, rep_best, acq_best = None, None, -1.0
    for i in range(search_n):
        action = comp_action_rand(arch)
        rep = comp_rep(arch, action)
        acq = kernel.acquisition(rep)
        if acq > acq_best:
            action_best, rep_best, acq_best = action, rep, acq
    return action_best, rep_best, acq_best
