import math
def schedule_ET(acc, N, N_max, std=False):
    if acc <= 1e-6:
        return N_max 
    
    if std: # 2 sqrt(acc * (1-acc))
        S_acc = 2 * math.sqrt(acc * (1-acc))
        A_g_peak = N
    else:
        S_acc =  2 * acc * (1-acc)
        A_g_peak = 0.5 * N
    
    A_g_current = 2 * N * S_acc

    delt_N = A_g_peak - A_g_current
    delt_N /= S_acc

    delt_N = min(math.ceil(delt_N), N_max)

    return delt_N