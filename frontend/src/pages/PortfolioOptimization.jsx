import React, { useState, useEffect } from 'react';
import { api } from '../services/api';
import Chart from 'react-apexcharts';
import Button from '../components/ui/Button';
import Spinner from '../components/ui/Spinner';
import { useToast } from '../components/ui/ToastProvider';

export default function PortfolioOptimization() {
  const [method, setMethod] = useState('markowitz');
  const [gamma, setGamma] = useState(1.0);
  const [positions, setPositions] = useState([]);
  const [weights, setWeights] = useState(null);
  const [loadingPositions, setLoadingPositions] = useState(true);
  const [loadingOptimize, setLoadingOptimize] = useState(false);
  const [frontierData, setFrontierData] = useState([]);
  const [loadingFrontier, setLoadingFrontier] = useState(false);
  const showToast = useToast();

  // 1️⃣ Pobierz aktualne pozycje
  useEffect(() => {
    api.get('/portfolio')
      .then(res => setPositions(res.data))
      .catch(() => showToast('Nie udało się wczytać portfela', 'error'))
      .finally(() => setLoadingPositions(false));
  }, []);

  // 2️⃣ Funkcja uruchamiająca optymalizację
  const runOptimization = async () => {
    setLoadingOptimize(true);
    try {
      const endpoint = method === 'markowitz'
        ? '/optimize/markowitz'
        : '/optimize/ai';
      const payload = method === 'markowitz' ? { gamma } : {};
      const res = await api.post(endpoint, payload);
      setWeights(res.data.weights);
    } catch {
      showToast('Błąd optymalizacji', 'error');
      setWeights(null);
    } finally {
      setLoadingOptimize(false);
    }
  };

  // 3️⃣ Gdy γ lub weights się zmieniają, pobierz dane frontiera
  useEffect(() => {
    if (method !== 'markowitz') {
      setFrontierData([]);
      return;
    }
    setLoadingFrontier(true);
    api.get(`/optimize/frontier?gamma_min=0&gamma_max=${gamma}&num_points=50`)
      .then(res => setFrontierData(res.data))
      .catch(() => showToast('Nie można wczytać efektywnej granicy', 'error'))
      .finally(() => setLoadingFrontier(false));
  }, [gamma, method]);

  // 4️⃣ Oblicz bieżące wagi z pozycji
  const currentWeights = React.useMemo(() => {
    if (!positions.length) return {};
    const vals = positions.map(p => p.total_quantity * p.current_price);
    const total = vals.reduce((s, v) => s + v, 0) || 1;
    return positions.reduce((acc, p, i) => {
      acc[p.symbol] = vals[i] / total;
      return acc;
    }, {});
  }, [positions]);

  // 5️⃣ Przygotuj dane do pie-charts
  const labels = Object.keys(currentWeights);
  const seriesCurrent = labels.map(sym => currentWeights[sym]);
  const seriesDesired = labels.map(sym => weights?.[sym] ?? 0);
  const pieOptions = {
    chart: { type: 'pie', background: '#1f2937' },
    theme: { mode: 'dark' },
    labels,
    legend: { position: 'bottom' }
  };

  // 6️⃣ Dane do scatter frontiera
  const scatterSeries = [{
    name: 'Efficient frontier',
    data: frontierData.map(p => ({ x: p.risk, y: p.expected_return }))
  }];
  const scatterOptions = {
    chart: {
      type: 'scatter',
      zoom: { enabled: true },
      background: '#1f2937'
    },
    theme: { mode: 'dark' },
    xaxis: { title: { text: 'Risk (σ)' } },
    yaxis: { title: { text: 'Expected Return (μ)' } },
    tooltip: {
      x: { formatter: x => x.toFixed(4) },
      y: { formatter: y => y.toFixed(4) }
    }
  };

  return (
    <div className="p-6 bg-surface rounded-lg shadow space-y-6">

      {/* Tytuł */}
      <h1 className="text-2xl font-bold">Porównanie stanu portfela</h1>

      {/* Tabs */}
      <nav className="flex space-x-4 border-b border-gray-700 pb-2">
        {['markowitz','ai'].map(m => (
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

      {/* Gamma only for Markowitz */}
      {method === 'markowitz' && (
        <div className="flex items-center space-x-2">
          <label className="text-gray-200">Gamma (risk aversion):</label>
          <input
            type="number"
            step="0.1" min="0" value={gamma}
            onChange={e => setGamma(parseFloat(e.target.value) || 0)}
            className="w-24 p-2 bg-gray-800 border border-gray-600 rounded text-white"
          />
        </div>
      )}

      {/* Run button */}
      <Button onClick={runOptimization} loading={loadingOptimize}>
        Optymalizuj portfel
      </Button>

      {/* Pokaż loader pozycji */}
      {loadingPositions && <Spinner className="mx-auto" />}

      {/* Pie charts */}
      {!loadingPositions && positions.length > 0 && (
        <div className="grid grid-cols-2 gap-4">
          <div className="bg-gray-900 p-4 rounded">
            <h2 className="text-lg font-medium text-gray-200 mb-2">
              Obecny stan
            </h2>
            <Chart options={pieOptions} series={seriesCurrent} type="pie" height={280}/>
          </div>
          <div className="bg-gray-900 p-4 rounded">
            <h2 className="text-lg font-medium text-gray-200 mb-2">
              Pożądany stan
            </h2>
            {weights
              ? <Chart options={pieOptions} series={seriesDesired} type="pie" height={280}/>
              : <p className="text-gray-400">Uruchom optymalizację</p>}
          </div>
        </div>
      )}

      {/* Scatter‐chart: Efficient Frontier */}
      {method === 'markowitz' && (
        <div className="bg-gray-900 p-4 rounded">
          <h2 className="text-lg font-medium text-gray-200 mb-2">
            Efficient Frontier (Return vs. Risk)
          </h2>
          {loadingFrontier
            ? <Spinner className="mx-auto" />
            : <Chart
                options={scatterOptions}
                series={scatterSeries}
                type="scatter"
                height={350}
              />}
        </div>
      )}

      {/* Brak pozycji */}
      {!loadingPositions && positions.length === 0 && (
        <p className="text-red-400">
          Brak aktywów w portfelu – dodaj transakcje, aby zobaczyć wykresy.
        </p>
      )}

    </div>
  );
}
