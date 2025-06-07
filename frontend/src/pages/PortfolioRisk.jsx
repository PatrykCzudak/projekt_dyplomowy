import React, { useState, forwardRef, useImperativeHandle } from 'react';
import { api } from '../services/api';
import Chart from 'react-apexcharts';

const PortfolioRisk = forwardRef((props, ref) => {
  const [analyses, setAnalyses] = useState([]);
  const [expanded, setExpanded] = useState({});

  const addAnalysis = (mode) => {
    const id = Date.now();
    setAnalyses(prev => [...prev, { id, mode, data: null, error: false }]);

    const url = mode === 'Classical'
      ? '/risk/portfolio/1/classical'
      : '/risk/portfolio/1/ai';

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

  function buildHistogram(data, bins = 30) {
    const min = Math.min(...data);
    const max = Math.max(...data);
    const binSize = (max - min) / bins || 1;
    const histogram = new Array(bins).fill(0);
    const binCenters = new Array(bins).fill(0);

    data.forEach(value => {
      let binIndex = Math.floor((value - min) / binSize);
      if (binIndex >= bins) binIndex = bins - 1;
      histogram[binIndex]++;
    });

    for (let i = 0; i < bins; i++) {
      binCenters[i] = min + (i + 0.5) * binSize;
    }

    return { histogram, binCenters };
  }

  return (
    <div>
      {analyses.map(a => (
        <div key={a.id} className="my-4 p-4 bg-gray-800 rounded">
          <h3 className="text-lg font-semibold mb-2">
            {a.mode === 'Classical'
              ? 'Classical Portfolio Analysis'
              : 'AI Portfolio Analysis'}
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
                </div>
              )}
              {expanded[a.id] && a.mode === 'Classical' && (() => {
                const { histogram, binCenters } = buildHistogram(a.data.returns);
                return (
                  <Chart
                    options={{
                      chart: { id: `portfolio-risk-${a.id}` },
                      xaxis: {
                        type: 'numeric',
                        labels: { style: { colors: '#ccc' } },
                        title: { text: 'Returns', style: { color: '#ccc' } }
                      },
                      yaxis: {
                        labels: { style: { colors: '#ccc' } },
                        title: { text: 'Frequency', style: { color: '#ccc' } }
                      },
                      annotations: {
                        xaxis: [
                          {
                            x: -a.data.VaR_parametric,
                            borderColor: '#FF0000',
                            label: {
                              borderColor: '#FF0000',
                              style: { color: '#fff', background: '#FF0000' },
                              text: `Parametric VaR`
                            }
                          },
                          {
                            x: -a.data.VaR_historical,
                            borderColor: '#00FF00',
                            label: {
                              borderColor: '#00FF00',
                              style: { color: '#000', background: '#00FF00' },
                              text: `Historical VaR`
                            }
                          },
                          {
                            x: -a.data.Expected_Shortfall,
                            borderColor: '#FFA500',
                            label: {
                              borderColor: '#FFA500',
                              style: { color: '#000', background: '#FFA500' },
                              text: `Expected Shortfall`
                            }
                          }
                        ]
                      },
                      theme: { mode: 'dark' },
                      grid: { borderColor: '#333' }
                    }}
                    series={[
                      {
                        name: 'Frequency',
                        data: binCenters.map((center, idx) => ({ x: center, y: histogram[idx] }))
                      }
                    ]}
                    type="bar"
                    height={300}
                  />
                );
              })()}
            </>
          ) : (
            <p className="text-gray-400">Loading…</p>
          )}
        </div>
      ))}
    </div>
  );
});

export default PortfolioRisk;
