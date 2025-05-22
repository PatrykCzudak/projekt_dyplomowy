import React, { useState } from 'react';
import { api } from '../services/api';
import Chart from 'react-apexcharts';

export default function PortfolioRisk() {
  const [analyses, setAnalyses] = useState([]);

  const addAnalysis = (type) => {
    const id = Date.now();
    const newAnalysis = { id, type, data: null, show: false };
    setAnalyses(prev => [...prev, newAnalysis]);

    const url = type === "Classical"
      ? "/risk/portfolio/1/classical"
      : "/risk/portfolio/1/ai";

    api.get(url)
      .then(res => {
        console.log(`[DEBUG] Response for ${type}:`, res.data);
        setAnalyses(prev =>
          prev.map(a => a.id === id ? { ...a, data: res.data } : a)
        );
      })
      .catch(err => {
        console.error(`[ERROR] Failed to fetch ${type} analysis:`, err);
        setAnalyses(prev =>
          prev.map(a => a.id === id ? { ...a, data: { error: "Failed to fetch data" } } : a)
        );
      });
  };

  const renderChart = (an) => {
    const chartOptions = {
      chart: { id: `risk-chart-${an.id}` },
      xaxis: {
        categories: an.type === "Classical"
          ? ["VaR Parametric", "VaR Historical", "Expected Shortfall"]
          : ["Prediction"]
      }
    };

    const chartData = an.type === "Classical"
      ? [
          an.data?.VaR_parametric ?? 0,
          an.data?.VaR_historical ?? 0,
          an.data?.Expected_Shortfall ?? 0
        ]
      : [an.data?.risk_prediction ?? 0];

    return (
      <Chart
        options={chartOptions}
        series={[{ name: "Value", data: chartData }]}
        type="bar"
        height={300}
      />
    );
  };

  return (
    <div>
      <button
        onClick={() => addAnalysis("Classical")}
        className="mr-2 bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded"
      >
        Add Classical
      </button>
      <button
        onClick={() => addAnalysis("AI")}
        className="bg-green-600 hover:bg-green-700 text-white px-4 py-2 rounded"
      >
        Add AI
      </button>

      {analyses.map(an => (
        <div key={an.id} className="my-4 p-4 bg-gray-800 rounded">
          <h3 className="text-lg font-semibold mb-2">{an.type} Portfolio Analysis</h3>
          {an.data ? (
            an.data.error ? (
              <p className="text-red-400">{an.data.error}</p>
            ) : (
              <>
                {an.type === "Classical" ? (
                  <div>
                    <p>VaR Param: {(an.data.VaR_parametric ?? 0).toFixed(4)}</p>
                    <p>VaR Hist: {(an.data.VaR_historical ?? 0).toFixed(4)}</p>
                    <p>Expected Shortfall: {(an.data.Expected_Shortfall ?? 0).toFixed(4)}</p>
                  </div>
                ) : (
                  <div>
                    <p>Risk Category: {an.data.risk_category}</p>
                    <p>Prediction: {(an.data.risk_prediction ?? 0).toFixed(4)}</p>
                  </div>
                )}
                <button
                  onClick={() =>
                    setAnalyses(prev =>
                      prev.map(a =>
                        a.id === an.id ? { ...a, show: !a.show } : a
                      )
                    )
                  }
                  className="mt-2 text-sm text-blue-300 hover:underline"
                >
                  {an.show ? "Hide" : "Show"} Details
                </button>
                {an.show && renderChart(an)}
              </>
            )
          ) : (
            <p className="text-gray-400">Loading...</p>
          )}
        </div>
      ))}
    </div>
  );
}
