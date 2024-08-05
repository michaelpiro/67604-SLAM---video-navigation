import gtsam
from ex6 import load

def get_relative_covariance(c1, c2, marginals):
    # marginals = gtsam.Marginals(graph, result)
    keys = gtsam.KeyVector()
    keys.append(c1)
    keys.append(c2)
    marginal_information = marginals.jointMarginalInformation(keys)
    c2_giving_c1 = marginal_information.at(c2,c2)
    return c2_giving_c1


if __name__ == '__main__':
    data = load('data/bundles')
    all_bundles = data[0]
    all_graphs = data[1]
    all_results = data[2]

    for i, bundle in all_bundles.items():
        graph = bundle['graph']
        result = bundle['result']
        marginals = gtsam.Marginals(graph, result)
        c1 = gtsam.symbol('c', 0)
        c2 = gtsam.symbol('c', 1)
        c2_giving_c1 = get_relative_covariance(c1, c2, marginals)
        print(c2_giving_c1)
        break
    print('done')