import { useState, useEffect } from "react";
import { api } from "../services/api";
import ReactApexChart from "react-apexcharts";

export default function ChartPage() {
  const [symbol, setSymbol] = useState("AAPL");
  const [period, setPeriod] = useState("1mo");
  const [chartType, setChartType] = useState("line");
  const [series, setSeries] = useState([]);

  useEffect(() => {
    api.get(`/assets/${symbol}/history?period=${period}`)
      .then(res => {
        const rawData = res.data.ohlc || [];
        if (chartType === "candlestick") {
          setSeries([{
            data: rawData
          }]);
        } else {
          setSeries([{
            name: symbol,
            data: rawData.map(d => ({ x: d.x, y: d.y[3] })) 
          }]);
        }
      })
      .catch(err => {
        console.error("Error loading data", err);
      });
  }, [symbol, period, chartType]);

  const options = {
    chart: {
      type: chartType,
      background: "#1f2937",
      toolbar: {
        show: true
      }
    },
    xaxis: {
      type: "category",
      labels: {
        style: {
          colors: "#ccc"
        }
      }
    },
    yaxis: {
      labels: {
        style: {
          colors: "#ccc"
        }
      }
    },
    theme: {
      mode: "dark"
    }
  };

  return (
    <div className="p-4">
      <h1 className="text-2xl font-semibold text-white mb-4">Asset Chart</h1>
      
      <div className="flex gap-4 mb-6">
        <input
          type="text"
          value={symbol}
          onChange={(e) => setSymbol(e.target.value.toUpperCase())}
          className="px-3 py-2 bg-gray-800 border border-gray-600 text-white rounded"
          placeholder="Enter symbol (e.g. AAPL)"
        />
        <select
          value={period}
          onChange={(e) => setPeriod(e.target.value)}
          className="px-3 py-2 bg-gray-800 border border-gray-600 text-white rounded"
        >
          <option value="1d">1d</option>
          <option value="5d">5d</option>
          <option value="1mo">1mo</option>
          <option value="3mo">3mo</option>
          <option value="6mo">6mo</option>
          <option value="1y">1y</option>
        </select>
        <select
          value={chartType}
          onChange={(e) => setChartType(e.target.value)}
          className="px-3 py-2 bg-gray-800 border border-gray-600 text-white rounded"
        >
          <option value="line">Line</option>
          <option value="candlestick">Candlestick</option>
        </select>
      </div>

      <div className="bg-gray-900 p-4 rounded shadow">
        <ReactApexChart options={options} series={series} type={chartType} height={400} />
      </div>
    </div>
  );
}
