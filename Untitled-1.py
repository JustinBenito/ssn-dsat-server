
        label_en=LabelEncoder()     
        feat.update(m)
        feat.update({'f0' : f0,
                'hnr' :hnr})
        features = pd.DataFrame.from_dict(feat)
        for i in features.columns:
            if features[i].dtype == object:
                print(i)
                features[i] = label_en.fit_transform(features[i])
        
        std_scaler = StandardScaler()
        min_max_scaler = MinMaxScaler(feature_range=(0,5))

        min_max_scaler.fit(features)
        features_scaled = min_max_scaler.transform(features)
        x = pd.DataFrame(features_scaled, columns=features.columns)
        loaded_model = pickle.load(open(os.path.join(basedir, app.config['model_folder'], 'finalized_model.sav'), 'rb'))
        res = loaded_model.predict(x)
        lb =   pickle.load(open(os.path.join(basedir, app.config['model_folder'], 'label_encoder.pkl'), 'rb')) 
        res1 = lb.inverse_transform(res)
        print(res)
        print("predicted class:: ",res1)

  <button onclick="playback()">Playback</button>