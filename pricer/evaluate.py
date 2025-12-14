import re
import math
from itertools import accumulate
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, r2_score
from tqdm.notebook import tqdm


# -------------------- Constants --------------------

WORKERS = 5
DEFAULT_SIZE = 200

GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"

COLOR_MAP = {
    "green": GREEN,
    "orange": YELLOW,
    "red": RED,
}


# -------------------- Core Class --------------------

class Tester:
    def __init__(self, predictor, data, size=DEFAULT_SIZE, workers=WORKERS, title=None):
        self.predictor = predictor
        self.data = data
        self.size = size
        self.workers = workers
        self.title = title or self._make_title(predictor)

        self.titles = []
        self.guesses = []
        self.truths = []
        self.errors = []
        self.colors = []

    # ---------- Helpers ----------

    @staticmethod
    def _make_title(fn):
        return (
            fn.__name__
            .replace("__", ".")
            .replace("_", " ")
            .title()
            .replace("Gpt", "GPT")
        )

    @staticmethod
    def _post_process(value):
        if not isinstance(value, str):
            return value

        value = value.replace("$", "").replace(",", "")
        match = re.search(r"[-+]?\d*\.\d+|\d+", value)
        return float(match.group()) if match else 0.0

    @staticmethod
    def _color_for(error, truth):
        if error < 40 or error / truth < 0.2:
            return "green"
        if error < 80 or error / truth < 0.4:
            return "orange"
        return "red"

    # ---------- Single datapoint ----------

    def _run_point(self, idx):
        dp = self.data[idx]

        guess = self._post_process(self.predictor(dp))
        truth = dp.price
        error = abs(guess - truth)

        title = dp.title[:40] + "..." if len(dp.title) > 40 else dp.title
        color = self._color_for(error, truth)

        return title, guess, truth, error, color

    # ---------- Charts ----------

    def _scatter_chart(self, title):
        df = pd.DataFrame(
            {
                "truth": self.truths,
                "guess": self.guesses,
                "title": self.titles,
                "error": self.errors,
                "color": self.colors,
            }
        )

        df["hover"] = [
            f"{t}\nGuess=${g:,.2f}  Actual=${y:,.2f}"
            for t, g, y in zip(df.title, df.guess, df.truth)
        ]

        max_val = float(max(df.truth.max(), df.guess.max()))

        fig = px.scatter(
            df,
            x="truth",
            y="guess",
            color="color",
            color_discrete_map={"green": "green", "orange": "orange", "red": "red"},
            labels={"truth": "Actual Price", "guess": "Predicted Price"},
            title=title,
            width=1000,
            height=800,
        )

        for trace in fig.data:
            mask = df.color == trace.name
            trace.customdata = df.loc[mask, ["hover"]].to_numpy()
            trace.hovertemplate = "%{customdata[0]}<extra></extra>"
            trace.marker.size = 6

        fig.add_trace(
            go.Scatter(
                x=[0, max_val],
                y=[0, max_val],
                mode="lines",
                line=dict(color="deepskyblue", dash="dash"),
                hoverinfo="skip",
                showlegend=False,
            )
        )

        fig.update_xaxes(range=[0, max_val])
        fig.update_yaxes(range=[0, max_val])
        fig.update_layout(showlegend=False)
        fig.show()

    def _error_trend_chart(self):
        n = len(self.errors)
        x = list(range(1, n + 1))

        sums = list(accumulate(self.errors))
        means = [s / i for s, i in zip(sums, x)]

        sq_sums = list(accumulate(e * e for e in self.errors))
        stds = [
            math.sqrt((sq / i) - mean**2) if i > 1 else 0
            for i, sq, mean in zip(x, sq_sums, means)
        ]

        ci = [1.96 * (sd / math.sqrt(i)) if i > 1 else 0 for i, sd in zip(x, stds)]
        upper = [m + c for m, c in zip(means, ci)]
        lower = [m - c for m, c in zip(means, ci)]

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=x + x[::-1],
                y=upper + lower[::-1],
                fill="toself",
                fillcolor="rgba(150,150,150,0.2)",
                line=dict(color="rgba(0,0,0,0)"),
                hoverinfo="skip",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=x,
                y=means,
                mode="lines",
                line=dict(width=3, color="firebrick"),
                customdata=ci,
                hovertemplate=(
                    "n=%{x}<br>"
                    "Avg Error=$%{y:,.2f}<br>"
                    "±95% CI=$%{customdata:,.2f}<extra></extra>"
                ),
            )
        )

        fig.update_layout(
            title=f"{self.title} Error: ${means[-1]:,.2f} ± ${ci[-1]:,.2f}",
            xaxis_title="Datapoints",
            yaxis_title="Average Absolute Error ($)",
            width=1000,
            height=360,
            template="plotly_white",
            showlegend=False,
        )

        fig.show()

    # ---------- Reporting ----------

    def _report(self):
        avg_error = sum(self.errors) / self.size
        mse = mean_squared_error(self.truths, self.guesses)
        r2 = r2_score(self.truths, self.guesses) * 100

        title = (
            f"{self.title} results<br>"
            f"<b>Error:</b> ${avg_error:,.2f} "
            f"<b>MSE:</b> {mse:,.0f} "
            f"<b>r²:</b> {r2:.1f}%"
        )

        self._error_trend_chart()
        self._scatter_chart(title)

    # ---------- Run ----------

    def run(self):
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            for title, guess, truth, error, color in tqdm(
                pool.map(self._run_point, range(self.size)),
                total=self.size,
            ):
                self.titles.append(title)
                self.guesses.append(guess)
                self.truths.append(truth)
                self.errors.append(error)
                self.colors.append(color)
                print(f"{COLOR_MAP[color]}${error:.0f} ", end="")

        self._report()


# -------------------- Public API --------------------

def evaluate(predictor, data, size=DEFAULT_SIZE, workers=WORKERS):
    Tester(predictor, data, size=size, workers=workers).run()
