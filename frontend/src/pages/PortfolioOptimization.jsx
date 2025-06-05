import React, { useState, useEffect } from 'react';
import { api } from '../services/api';
import Chart from 'react-apexcharts';
import Button from '../components/ui/Button';
import Spinner from '../components/ui/Spinner';
import { useToast } from '../components/ui/ToastProvider';
import { FaInfoCircle } from 'react-icons/fa';

export default function PortfolioOptimization() {
  const [method, setMethod] = useState('markowitz');
  const [gamma, setGamma] = useState(5.0);
  const [period, setPeriod] = useState('5y');
  const [positions, setPositions] = useState([]);
  const [weights, setWeights] = useState(null);
  const [muValues, setMuValues] = useState(null);
  const [loadingPositions, setLoadingPositions] = useState(true);
  const [loadingOptimize, setLoadingOptimize] = useState(false);
  const [frontierData, setFrontierData] = useState([]);
  const [loadingFrontier, setLoadingFrontier] = useState(false);
  const [cloudData, setCloudData] = useState([]);
  const [loadingCloud, setLoadingCloud] = useState(false);
  const showToast = useToast();

  // Fetch current positions
  useEffect(() => {
    api.get('/portfolio')
      .then(res => setPositions(res.data))
      .catch(() => showToast('Failed to load portfolio.', 'error'))
      .finally(() => setLoadingPositions(false));
  }, []);

  // Run optimization
  const runOptimization = async () => {
    setLoadingOptimize(true);
    try {
      const endpoint = method === 'markowitz'
        ? '/optimize/markowitz'
        : '/optimize/markowitz-ai';
      const payload = method === 'markowitz'
        ? { gamma, period }
        : { gamma, period, top_n: 5 };
      const res = await api.post(endpoint, payload);
      setWeights(res.data.weights);
      setMuValues(res.data.mu || null);
    } catch {
      showToast('Optimization failed.', 'error');
      setWeights(null);
      setMuValues(null);
    } finally {
      setLoadingOptimize(false);
    }
  };

  // Efficient frontier
  useEffect(() => {
    if (method !== 'markowitz') {
      setFrontierData([]);
      return;
    }
    setLoadingFrontier(true);
    api.get(`/optimize/frontier?gamma_min=0&gamma_max=${gamma}&num_points=50&period=${period}`)
      .then(res => setFrontierData(res.data))
      .catch(() => showToast('Failed to load efficient frontier.', 'error'))
      .finally(() => setLoadingFrontier(false));
  }, [gamma, method, period]);

  // Portfolio cloud
  useEffect(() => {
    if (method !== 'markowitz') {
      setCloudData([]);
      return;
    }
    setLoadingCloud(true);
    api.get(`/optimize/cloud?num_points=3000&period=${period}`)
      .then(res => setCloudData(res.data))
      .catch(() => showToast('Failed to load portfolio cloud.', 'error'))
      .finally(() => setLoadingCloud(false));
  }, [method, period]);

  // Calculate current weights
  const currentWeights = React.useMemo(() => {
    if (!positions.length) return {};
    const vals = positions.map(p => p.total_quantity * p.current_price);
    const total = vals.reduce((s, v) => s + v, 0) || 1;
    return positions.reduce((acc, p, i) => {
      acc[p.symbol] = vals[i] / total;
      return acc;
    }, {});
  }, [positions]);

  // Pie chart data
  const labels = Object.keys(currentWeights);
  const seriesCurrent = labels.map(sym => currentWeights[sym]);
  const seriesDesired = labels.map(sym => weights?.[sym] ?? 0);
  const pieOptions = {
    chart: { type: 'pie', background: '#1f2937' },
    theme: { mode: 'dark' },
    labels,
    legend: { position: 'bottom' }
  };

  // Efficient frontier chart
  const frontierOptions = {
    chart: {
      background: '#1f2937',
      zoom: { enabled: true },
      toolbar: { show: true }
    },
    theme: { mode: 'dark' },
    stroke: {
      width: [0, 2, 0],
      curve: 'smooth'
    },
    markers: {
      size: [3, 0, 5],
      colors: ['#3b82f6', '#00E396', '#FEB019'],
      strokeColors: '#1f2937',
      strokeWidth: 2,
      hover: {
        size: 6
      }
    },
    xaxis: {
      title: { text: 'Risk (σ)' },
      labels: {
        formatter: (val) => val.toFixed(4)
      }
    },
    yaxis: {
      title: { text: 'Expected Return (μ)' },
      labels: {
        formatter: (val) => val.toFixed(4)
      }
    },
    tooltip: {
      x: { formatter: (x) => `Risk: ${x.toFixed(4)}` },
      y: { formatter: (y) => `Return: ${y.toFixed(4)}` }
    },
    grid: {
      borderColor: '#374151',
      row: {
        colors: ['#1f2937', 'transparent'],
        opacity: 0.5
      }
    },
    legend: {
      position: 'top',
      labels: { colors: ['#fff'] }
    }
  };

  const frontierSeries = [
    {
      name: 'Portfolio Cloud',
      type: 'scatter',
      data: cloudData.map(p => ({ x: p.risk, y: p.expected_return }))
    },
    {
      name: 'Efficient Frontier (Line)',
      type: 'line',
      data: frontierData.map(p => ({ x: p.risk, y: p.expected_return }))
    },
    {
      name: 'Frontier Points',
      type: 'scatter',
      data: frontierData.map(p => ({ x: p.risk, y: p.expected_return }))
    }
  ];

  return (
    <div className="p-6 bg-surface rounded-lg shadow space-y-6">
      <h1 className="text-2xl font-bold">Portfolio Optimization</h1>

      {/* Tabs */}
      <nav className="flex space-x-4 border-b border-gray-700 pb-2">
        {['markowitz', 'ai'].map(m => (
          <button
            key={m}
            onClick={() => setMethod(m)}
            className={`py-2 px-4 font-medium ${
              method === m
                ? 'border-b-2 border-primary text-primary'
                : 'text-gray-400 hover:text-gray-200'
            }`}
          >
            {m === 'markowitz' ? 'Markowitz' : 'AI'}
          </button>
        ))}
      </nav>

      {/* Controls */}
      {(method === 'markowitz' || method === 'ai') && (
        <div className="flex items-center space-x-4 flex-wrap">
          <div className="flex items-center space-x-2">
            <label className="text-gray-200 flex items-center space-x-1">
              Gamma (risk aversion):
              <FaInfoCircle
                className="text-blue-400 cursor-pointer"
                title={`Low gamma (0.1–1) — aggressive portfolio.\nMedium gamma (2–5) — balanced portfolio.\nHigh gamma (8–10) — defensive portfolio.`}
              />
            </label>
            <input
              type="number"
              step="0.1"
              min="0"
              value={gamma}
              onChange={e => setGamma(parseFloat(e.target.value) || 0)}
              className="w-24 p-2 bg-gray-800 border border-gray-600 rounded text-white"
            />
          </div>
          <div className="flex items-center space-x-2">
            <label className="text-gray-200">Historical period:</label>
            <select
              value={period}
              onChange={e => setPeriod(e.target.value)}
              className="p-2 bg-gray-800 border border-gray-600 rounded text-white"
            >
              <option value="1y">1 year</option>
              <option value="3y">3 years</option>
              <option value="5y">5 years</option>
              <option value="10y">10 years</option>
              <option value="max">max</option>
            </select>
          </div>
        </div>
      )}

      {/* Run button */}
      <Button onClick={runOptimization} loading={loadingOptimize}>
        Optimize Portfolio
      </Button>

      {loadingPositions && <Spinner className="mx-auto w-8 h-8" />}

      {/* Pie charts */}
      {!loadingPositions && positions.length > 0 && (
        <div className="grid grid-cols-2 gap-4">
          <div className="bg-gray-900 p-4 rounded">
            <h2 className="text-lg font-medium text-gray-200 mb-2">
              Current State
            </h2>
            <Chart options={pieOptions} series={seriesCurrent} type="pie" height={280} />
          </div>
          <div className="bg-gray-900 p-4 rounded">
            <h2 className="text-lg font-medium text-gray-200 mb-2">
              Desired State
            </h2>
            {weights
              ? <Chart options={pieOptions} series={seriesDesired} type="pie" height={280} />
              : <p className="text-gray-400">Run optimization to see results.</p>}
          </div>
        </div>
      )}

      {/* Predicted Returns */}
      {muValues && (
        <div className="bg-gray-900 p-4 rounded">
          <h2 className="text-lg font-medium text-gray-200 mb-2">
            Predicted Returns (μ)
          </h2>
          <table className="min-w-full text-white">
            <thead>
              <tr>
                <th className="px-4 py-2 text-left">Symbol</th>
                <th className="px-4 py-2 text-left">μ</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(muValues).map(([symbol, mu]) => (
                <tr key={symbol} className="border-t border-gray-700">
                  <td className="px-4 py-2">{symbol}</td>
                  <td className="px-4 py-2">{mu.toFixed(4)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Efficient Frontier */}
      {method === 'markowitz' && (
        <div className="bg-gray-900 p-4 rounded">
          <h2 className="text-lg font-medium text-gray-200 mb-2 flex items-center space-x-2">
            <span>Efficient Frontier (Return vs. Risk)</span>
            <FaInfoCircle
              className="text-blue-400 cursor-pointer"
              title={`This chart shows:\n- Portfolio Cloud: Random portfolios with different risk-return profiles.\n- Efficient Frontier (Line): The best possible trade-off between risk and return.\n- Frontier Points: Individual optimal solutions on the frontier.`}
            />
          </h2>
          {loadingFrontier || loadingCloud
            ? <Spinner className="mx-auto w-8 h-8" />
            : <Chart
                options={frontierOptions}
                series={frontierSeries}
                type="line"
                height={400}
              />}
        </div>
      )}

      {!loadingPositions && positions.length === 0 && (
        <p className="text-red-400">
          No assets in the portfolio — please add transactions to view charts.
        </p>
      )}
    </div>
  );
}
