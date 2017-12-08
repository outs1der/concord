#===============================================================
# Tools that return parameters for
# different concord versions (con_ver)
#===============================================================

def get_weights(con_ver):
    """========================================================
    Returns pre-defined lhood weights, based on con_ver
    ========================================================"""
    tdelwts = {1:1., 2:2.5e3, 3:100., 4:2.5e3, 5:2.5e3, 6:500}   # tdel weights for fitting (for different con_ver)
    fluxwts = {1:1., 2:1., 3:1., 4:1., 5:100., 6:100}
    return {'tdelwt':tdelwts[con_ver], 'fluxwt':fluxwts[con_ver]}


def get_disc_model(con_ver):
    """========================================================
    Returns pre-defined disc model of given con_ver
    ========================================================"""
    discs = {1:'he16_a', 2:'he16_a', 3:'he16_a', 4:'he16_d', 5:'he16_d',
                6:'he16_a'}
    return discs[con_ver]


def get_lhood_floor(con_ver):
    """========================================================
    Returns lhood floor for plotting
    ========================================================"""
    floors = {2:-5e4, 3:-2e4, 4:-5e4, 5:0.0, 6:-5e4}    # lower cutoff for lhood plotting
    return floors[con_ver]
