import React, { useState, useEffect } from 'react';
import { api } from '../services/api';
import ReactApexChart from 'react-apexcharts';
import Spinner from '../components/ui/Spinner';
import { useToast } from '../components/ui/ToastProvider';
import TickerSelect from '../components/ui/TickerSelect';
import tickers from '../tickers.json';

export default function ChartPage() {
  const [symbol, setSymbol] = useState('AAPL');
  const [period, setPeriod] = useState('1mo');
  const [chartType, setChartType] = useState('line');
  const [historicalData, setHistoricalData] = useState([]);
  const [forecastData, setForecastData] = useState([]);
  const [loading, setLoading] = useState(false);
  const showToast = useToast();

  const tickerOptions = tickers.map(ticker => ({
    label: ticker,
    value: ticker
  }));

  useEffect(() => {
    const timeout = setTimeout(() => {
      setLoading(true);
      api
        .get(`/assets/${symbol}/history?period=${period}`)
        .then(res => {
          const raw = res.data.ohlc || [];
          const data = raw.map(d => ({
            date: d.x,
            close: d.y[3]
          }));
          setHistoricalData(data);
        })
        .catch(err => {
          console.error('Error loading chart data', err);
          showToast(`Błąd ładowania danych dla ${symbol}`, 'error');
        })
        .finally(() => setLoading(false));
    }, 500); // 500 ms

    return () => clearTimeout(timeout);
  }, [symbol, period]);

  useEffect(() => {
    const timeout = setTimeout(() => {
      api
        .get(`/price/asset/${symbol}/forecast`)
        .then(res => {
          setForecastData(res.data.forecast);
        })
        .catch(err => {
          console.error('Błąd ładowania prognozy', err);
          showToast(`Błąd ładowania prognozy dla ${symbol}`, 'error');
        });
    }, 500); // 500 ms

    return () => clearTimeout(timeout);
  }, [symbol]);

  const historySeries = chartType === 'candlestick'
    ? []
    : [{
        name: 'Historyczne',
        data: historicalData.map(d => ({ x: d.date, y: d.close }))
      }];

  const forecastSeries = forecastData.length > 0 && historicalData.length > 0 ? [
    {
      name: 'Prognoza (średnia)',
      data: [
        {
          x: historicalData[historicalData.length - 1].date,
          y: historicalData[historicalData.length - 1].close
        },
        ...forecastData.map(d => ({ x: d.date, y: d.mean }))
      ]
    },
    {
      name: 'Prognoza (górny)',
      data: [
        {
          x: historicalData[historicalData.length - 1].date,
          y: historicalData[historicalData.length - 1].close
        },
        ...forecastData.map(d => ({ x: d.date, y: d.mean + 2 * d.std }))
      ]
    },
    {
      name: 'Prognoza (dolny)',
      data: [
        {
          x: historicalData[historicalData.length - 1].date,
          y: historicalData[historicalData.length - 1].close
        },
        ...forecastData.map(d => ({ x: d.date, y: d.mean - 2 * d.std }))
      ]
    }
  ] : [];

  const options = {
    chart: { type: chartType, toolbar: { show: true } },
    theme: { mode: 'dark' },
    stroke: { curve: 'smooth' },
    grid: { borderColor: '#333' },
    xaxis: {
      type: 'datetime',
      tickAmount: 8,
      labels: {
        style: { colors: '#ccc' },
        rotate: -45,
        hideOverlappingLabels: true,
        rotateAlways: true,
        formatter: value => {
          const d = new Date(value);
          return `${String(d.getDate()).padStart(2, '0')}/${String(d.getMonth() + 1).padStart(2, '0')}`;
        }
      },
      axisTicks: { show: false },
      axisBorder: { show: false }
    },
    yaxis: { labels: { style: { colors: '#ccc' } } },
  };

  const handleTickerChange = (selectedOption) => {
    if (selectedOption && selectedOption.value) {
      setSymbol(selectedOption.value.toUpperCase());
    }
  };

  return (
    <div className="bg-gray-800 p-4">
      <div className="flex flex-wrap items-center gap-4 mb-4">
        <div className="w-64">
          <TickerSelect
            value={tickerOptions.find(opt => opt.value === symbol)}
            onChange={handleTickerChange}
            options={tickerOptions}
            placeholder="Choose ticker..."
          />
        </div>
        <select
          value={period}
          onChange={e => setPeriod(e.target.value)}
          className="px-3 py-2 bg-gray-700 border border-gray-600 text-white rounded"
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
          onChange={e => setChartType(e.target.value)}
          className="px-3 py-2 bg-gray-700 border border-gray-600 text-white rounded"
        >
          <option value="line">Line</option>
          <option value="candlestick">Candlestick</option>
        </select>
      </div>

      <div className="w-full h-96 bg-gray-900 rounded shadow overflow-hidden">
        {loading ? (
          <div className="flex items-center justify-center h-full">
            <Spinner className="w-12 h-12 text-primary" />
          </div>
        ) : (
          <ReactApexChart
            options={options}
            series={[...historySeries, ...forecastSeries]}
            type={chartType}
            width="100%"
            height="100%"
          />
        )}
      </div>
    </div>
  );
}
