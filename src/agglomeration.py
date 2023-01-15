import numpy as np
from sklearn.cluster import FeatureAgglomeration, DBSCAN, KMeans, AgglomerativeClustering, Birch, MeanShift

if __name__ == '__main__':
    # vectors = [[[0.17071414552044223, 0.2821446285432688, 0.20880409091773383],
    #             [0.172843504190474, 0.3111324965683954, 0.2383623680117088],
    #             [0.13272987373844436, 0.2570366957594332, 0.20221130870650977]],
    #            [[0.0870359337014771, 0.12829541611554718, 0.08123791430571248],
    #             [0.13033465266568767, 0.19950155838203049, 0.12797499049060806],
    #             [0.09938533260548105, 0.1363837162385229, 0.08331506909295665]],
    #            [[0.17421028540797495, 0.3721556066952534, 0.4446739246606819],
    #             [0.07861074397418491, 0.16403232887551394, 0.19453076402554315],
    #             [0.01412152055115276, 0.02867441997958605, 0.03370117913565025]]]

    # vectors = [[0.17071414552044223, 0.2821446285432688, 0.20880409091773383],
    #            [0.0870359337014771, 0.12829541611554718, 0.08123791430571248],
    #            [0.17421028540797495, 0.3721556066952534, 0.4446739246606819]]

    vectors = [[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
               [[0.5037242398263571,
                 1.0037167450906808,
                 0.9763104525753911,
                 0.4468210472372068,
                 0.09018689425671127,
                 0.007583706151895067,
                 0.00025532375706602663,
                 3.3531227710736447e-06]],
               [[0.48395463615060674,
                 1.012404752383287,
                 1.0449911187949357,
                 0.4993851961999395,
                 0.10275943177411903,
                 0.008650109598183315,
                 0.0002885004166100784,
                 3.735451330367279e-06]],
               [[0.3242354002530576,
                 1.0307853566572793,
                 1.5892008417942218,
                 1.226554717833095,
                 0.48961070857388933,
                 0.09778982361864184,
                 0.008905611478226367,
                 0.00033912320351559184]],
               [[0.2984945276561274,
                 0.38973227300043994,
                 0.2170629209330689,
                 0.1712774011610254,
                 0.5049082770638046,
                 0.8754209856834938,
                 1.0417749673506385,
                 1.1625600320966787]],
               [[0.3706739967069536,
                 0.6723763566558445,
                 0.764591764817673,
                 0.6415414812791835,
                 0.3231249506531429,
                 0.07571940907614773,
                 0.007340678234411756,
                 0.00028102643678187085]],
               [[0.3706739967069536,
                 0.6723763566558445,
                 0.764591764817673,
                 0.6415414812791835,
                 0.3231249506531429,
                 0.07571940907614773,
                 0.007340678234411756,
                 0.00028102643678187085]],
               [[0.556985811968761,
                 0.9910707639406162,
                 1.0739315350039236,
                 0.7672393198809959,
                 0.28962791578770586,
                 0.04818650978542053,
                 0.003287666216414827,
                 8.919639309984165e-05]],
               [[0.2984863650615487,
                 0.38920268815602643,
                 0.20384754660349777,
                 0.042678268103415115,
                 0.0035364450103539716,
                 0.00011459576289465657,
                 1.435448393416525e-06,
                 6.882681558700004e-09]],
               [[0.007055472741704352,
                 0.07189031711756716,
                 0.28820141201332294,
                 0.45878827297060015,
                 0.2910472617963543,
                 0.07332240788240538,
                 0.007268464925784868,
                 0.00028017806930998235]],
               [[0.36362668651238406,
                 0.6010155990133297,
                 0.4896006093787544,
                 0.3109599281032739,
                 0.5219018377141109,
                 0.745776320873865,
                 0.44965779569737085,
                 0.10794036371896497]],
               [[0.5621683465202194,
                 0.9233118037507021,
                 0.7090846989023016,
                 0.2661330776681874,
                 0.04778772368775234,
                 0.003819507783838741,
                 0.0001265556832334443,
                 1.658515957261051e-06]],
               [[0.2110370951497374,
                 0.3948154855469677,
                 0.7094282451657716,
                 0.9218014286624199,
                 0.5484562278284266,
                 0.1327744899670783,
                 0.012744183609648106,
                 0.0004782496768501658]],
               [[0.3162479136592087,
                 0.755371868183812,
                 1.2146393139627705,
                 1.3154707917483284,
                 0.7971770808329466,
                 0.22334162900846172,
                 0.026105182744340844,
                 0.001215637110928082]],
               [[0.66792739904779,
                 1.6679842851533018,
                 1.9777649496069114,
                 1.0628053880801818,
                 0.2447410549971734,
                 0.023109663270287968,
                 0.0008679342220107684,
                 1.2700628201410022e-05]],
               [[0.5254926140577028,
                 1.1135722774020373,
                 1.2358445552850108,
                 0.6578744353802962,
                 0.15287859147246136,
                 0.014596751182852223,
                 0.0005529109927403944,
                 8.133149330169705e-06]],
               [[0.3499453497952195,
                 0.9819758250941513,
                 1.226122957115971,
                 0.6672909781490156,
                 0.15364078019463656,
                 0.014525117260651484,
                 0.0005495793475360837,
                 8.15708883876919e-06]],
               [[0.2984863650615487,
                 0.38920268815602643,
                 0.20384754660349777,
                 0.042678268103415115,
                 0.0035364450103539716,
                 0.00011459576289465657,
                 1.435448393416525e-06,
                 6.882681558700004e-09]]]

    prepared_vectors = []
    for vector in vectors:
      prepared_vectors.append(vector[0])
    prepared_vectors = np.array(prepared_vectors)

    labels = {}

    kmeans = KMeans(n_clusters=6)
    kmeans.fit(prepared_vectors)
    labels["kmeans"] = kmeans.labels_

    agglo = AgglomerativeClustering(n_clusters=4)
    agglo.fit(prepared_vectors)
    labels["agglomeration"] = agglo.labels_

    feature_agglo = FeatureAgglomeration(n_clusters=4)
    feature_agglo.fit_transform(prepared_vectors)
    labels["feature agglomeration"] = feature_agglo.labels_

    db = DBSCAN(eps=0.8, min_samples=2)
    db.fit(prepared_vectors)
    labels["dbscan"] = db.labels_

    birch = Birch()
    birch.fit(prepared_vectors)
    labels["birch"] = birch.labels_

    mean_shift = MeanShift()
    mean_shift.fit(prepared_vectors)
    labels["mean_shift"] = mean_shift.labels_

    print('x')