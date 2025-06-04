import React, { useState, forwardRef, useImperativeHandle } from 'react';
import { api } from '../services/api';
import Chart from 'react-apexcharts';
import TickerSelect from '../components/ui/TickerSelect';
import tickers from '../tickers.json';

const AssetRisk = forwardRef((props, ref) => {
  const [symbol, setSymbol] = useState('');
  const [analyses, setAnalyses] = useState([]);
  const [expanded, setExpanded] = useState({});

  const tickerOptions = tickers.map(ticker => ({
    label: ticker,
    value: ticker
  }));

  const addAnalysis = (mode) => {
    const id = Date.now();
    setAnalyses(prev => [...prev, { id, mode, symbol, data: null, error: false }]);

    const url = mode === 'Classical'
      ? `/risk/asset/${symbol}/classical`
      : `/risk/asset/${symbol}/ai`;

    api.get(url)
      .then(res => {
        setAnalyses(prev =>
          prev.map(a => a.id === id ? { ...a, data: res.data } : a)
        );
      })
      .catch(() => {
        setAnalyses(prev =>
          prev.map(a => a.id === id ? { ...a, error: true } : a)
        );
      });
  };

  useImperativeHandle(
    ref,
    () => ({ addAnalysis }),
    [addAnalysis]
  );

  const toggle = id => {
    setExpanded(e => ({ ...e, [id]: !e[id] }));
  };

  const handleTickerChange = (selectedOption) => {
    if (selectedOption && selectedOption.value) {
      setSymbol(selectedOption.value.toUpperCase());
    }
  };

  return (
    <div>
      <div className="w-64">
        <TickerSelect
          value={tickerOptions.find(opt => opt.value === symbol)}
          onChange={handleTickerChange}
          options={tickerOptions}
          placeholder="Choose ticker..."
        />
      </div>

      {analyses.map(a => (
        <div key={a.id} className="my-4 p-4 bg-gray-800 rounded">
          <h3 className="text-lg font-semibold mb-2">
            {a.mode === 'Classical'
              ? 'Classical Asset Analysis'
              : 'AI Asset Analysis'} – {a.symbol}
          </h3>

          {a.error ? (
            <p className="text-red-400">Failed to fetch data</p>
          ) : a.data ? (
            <>
              {a.mode === 'Classical' ? (
                <div className="mb-4 space-y-1 text-gray-200">
                  <p>VaR Parametric: {a.data.VaR_parametric.toFixed(4)}</p>
                  <p>VaR Historical: {a.data.VaR_historical.toFixed(4)}</p>
                  <p>Expected Shortfall: {a.data.Expected_Shortfall.toFixed(4)}</p>
                  <button
                    onClick={() => toggle(a.id)}
                    className="
                      inline-flex items-center justify-center
                      text-xs font-semibold
                      bg-primary text-white
                      px-3 py-1
                      rounded-full
                      hover:bg-primary-light
                      focus:outline-none focus:ring-2 focus:ring-primary/50
                      transition
                    "
                  >
                    {expanded[a.id] ? 'Hide details' : 'Show more'}
                  </button>                
                </div>
              ) : (
                <div className="mb-4 space-y-1 text-gray-200">
                  <p>Risk Category: {a.data.risk_category}</p>
                  <p>Prediction: {a.data.risk_prediction.toFixed(4)}</p>
                  <button
                    onClick={() => toggle(a.id)}
                    className="
                      inline-flex items-center justify-center
                      text-xs font-semibold
                      bg-primary text-white
                      px-3 py-1
                      rounded-full
                      hover:bg-primary-light
                      focus:outline-none focus:ring-2 focus:ring-primary/50
                      transition
                    "
                  >
                    {expanded[a.id] ? 'Hide details' : 'Show more'}
                  </button>                
                </div>
              )}
              {expanded[a.id] && (
                <Chart
                  options={{
                    chart: { id: `asset-risk-${a.id}` },
                    xaxis: {
                      categories: a.mode === 'Classical'
                        ? ['VaR Parametric','VaR Historical','Expected Shortfall']
                        : ['Prediction']
                    },
                    theme: { mode: 'dark' },
                    stroke: { curve: 'smooth' },
                    yaxis: { labels: { style: { colors: '#ccc' } } },
                    grid: { borderColor: '#333' }
                  }}
                  series={[{
                    name: 'Value',
                    data: a.mode === 'Classical'
                      ? [
                          a.data.VaR_parametric,
                          a.data.VaR_historical,
                          a.data.Expected_Shortfall
                        ]
                      : [a.data.risk_prediction]
                  }]}
                  type="bar"
                  height={300}
                />
              )}
            </>
          ) : (
            <p className="text-gray-400">Loading…</p>
          )}
        </div>
      ))}
    </div>
  );
});

export default AssetRisk;
