import { useState, useEffect } from 'react';
import Chart from 'react-apexcharts';
import { api } from '../services/api';

export default function HistoricalChart() {
  const [symbol, setSymbol] = useState('AAPL');
  const [period, setPeriod] = useState('1y'); // np. 1d, 5d, 1mo, 3mo, 1y
  const [series, setSeries] = useState([]);

  useEffect(() => {
    if (!symbol) return;

    api.get(`/assets/${symbol}/history?period=${period}`)
      .then(res => {
        const data = res.data.historical_prices;
        const sorted = Object.entries(data).sort(([a], [b]) => new Date(a) - new Date(b));
        const chartData = sorted.map(([date, price]) => ({
          x: date,
          y: price
        }));
        setSeries([{ name: symbol, data: chartData }]);
      })
      .catch(err => {
        console.error("Failed to fetch historical prices", err);
        setSeries([]);
      });
  }, [symbol, period]);

  return (
    <div className="text-gray-100 p-6">
      <h1 className="text-2xl font-semibold mb-4">Historical Price Chart</h1>

      <div className="flex items-center gap-4 mb-4">
        <input
          className="bg-gray-900 border border-gray-700 rounded px-3 py-2 text-sm text-white"
          placeholder="Enter asset symbol"
          value={symbol}
          onChange={(e) => setSymbol(e.target.value.toUpperCase())}
        />

        <select
          className="bg-gray-900 border border-gray-700 rounded px-3 py-2 text-sm text-white"
          value={period}
          onChange={(e) => setPeriod(e.target.value)}
        >
          <option value="5d">5d</option>
          <option value="1mo">1mo</option>
          <option value="3mo">3mo</option>
          <option value="6mo">6mo</option>
          <option value="1y">1y</option>
        </select>
      </div>

      <div className="bg-gray-800 p-4 rounded shadow">
        {series.length > 0 ? (
          <Chart
            type="line"
            height={400}
            series={series}
            options={{
              chart: {
                id: "price-chart",
                toolbar: { show: true },
                zoom: { enabled: true }
              },
              xaxis: {
                type: 'datetime',
                labels: { style: { colors: '#cbd5e1' } }
              },
              yaxis: {
                labels: { style: { colors: '#cbd5e1' } }
              },
              theme: {
                mode: 'dark'
              },
              tooltip: {
                theme: 'dark'
              }
            }}
          />
        ) : (
          <p>No data available</p>
        )}
      </div>
    </div>
  );
}
