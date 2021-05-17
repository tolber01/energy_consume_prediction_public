class WintersNoTrend:
    def __init__(self, s):
        self.s = s # период сезонности
        self.a = None
        self.theta = None
        self.n = None
    
    def fit(self, train, alpha):
        y = train["X"]
        self.n = train.shape[0]
    
        self.a = np.zeros(self.n)
        self.theta = np.zeros(self.n)
        m = y.mean()
        
        for t in range(self.s):
            self.theta[t] = y[t]
            self.a[t] = m
        
        for t in range(s, self.n):
            self.a[t] = alpha[0] * (y[t] / self.theta[t - self.s]) + (1 - alpha[0]) * self.a[t - 1]
            self.theta[t] = alpha[1] * (y[t] / self.a[t]) + (1 - alpha[1]) * self.theta[t - self.s]
    
    def predict(self, d=1):
        y_predicted = np.zeros(d)
        
        for t in range(self.n, self.n + d):
            y_predicted[t - self.n] = self.a[t - d] * self.theta[t - d + (d % self.s) - self.s]
        
        return y_predicted


def optimize_alpha(alpha, s, train_data, d):
    train_data, test_data = train_data[:-d,:], train_data[d:,:]
    winters_model = WintersNoTrend(s)
    winters_model.fit(train_data, alpha)
    predicted_data = winters_model.predict(d)
    
    return ((predicted_data - test_data["X"].to_numpy())**2).sum()


def error(real, predicted, mean=True):
    real_values = real["X"].to_numpy()
    errors = np.abs(real_values - predicted) / np.abs(real_values)
    return np.mean(errors) if mean else np.median(errors)

def draw_results(real, predicted):
    real_values = real["X"].to_numpy()
    results_predict = pd.DataFrame(
        data={
            "real": real_values,
            "predicted": predicted
        }
    )
    results_predict.plot(figsize=(17, 4))

def test_model(model, real_data, d=1, draw_plot=True, mean=True, **kwargs):
    predicted_values = model.predict(d=d, **kwargs)
    
    if draw_plot:
        draw_results(real_data, predicted_values)
    
    real_sum = real_data["X"].sum()
    predicted_sum = predicted_values.sum()
    
    return error(real_data, predicted_values, mean), abs(real_sum - predicted_sum) / abs(real_sum)


if __name__ == "__main__":
    alpha = np.array([0, 1])
    s = 1 * 7 * 24
    d = min(1 * 31 * 24, data[predict_month].shape[0])
    real_next_values = data[predict_month].iloc[:d, :]

    winters_model = WintersNoTrend(s)
    winters_model.fit(train_data, alpha)

    errors = test_model(winters_model, real_next_values, d=d, mean=False)
    print("Errors - per hour: {}, month sum: {}".format(*errors))
