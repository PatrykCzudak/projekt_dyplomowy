import React, { useState, useEffect } from 'react';
import { TrendingUp, TrendingDown } from 'lucide-react';
import { api } from '../services/api';

export default function PortfolioTable({ refreshTrigger, onRowClick }) {
  const [portfolio, setPortfolio] = useState([]);
  const [loading, setLoading] = useState(true);

  const SKELETON_ROWS = 5;

  useEffect(() => {
    setLoading(true);
    api.get('/portfolio/')
      .then(res => setPortfolio(res.data))
      .catch(err => console.error('Failed to fetch portfolio data:', err))
      .finally(() => setLoading(false));
  }, [refreshTrigger]);

  return (
    <div>
      <h2 className="text-xl font-semibold mb-4">Portfolio Overview</h2>
      <div className="overflow-x-auto">
        <table className="min-w-full border-collapse">
          <thead>
            <tr className="border-b border-gray-700">
              {['Asset','Quantity','Price','Value','Change'].map(h => (
                <th key={h} className="py-2 px-4 text-left text-xs font-semibold text-gray-400 uppercase">
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {loading
              ? Array.from({ length: SKELETON_ROWS }).map((_, i) => (
                  <tr key={i} className={i % 2 !== 0 ? 'bg-gray-900' : ''}>
                    {Array.from({ length: 5 }).map((__, j) => (
                      <td key={j} className="py-3 px-4">
                        <div className="h-4 bg-gray-700 rounded animate-pulse" />
                      </td>
                    ))}
                  </tr>
                ))
              : portfolio.length > 0
                ? portfolio.map((item, idx) => {
                    const value = item.total_quantity * item.current_price;
                    const percentChange = ((item.current_price - item.average_price) / item.average_price) * 100;
                    const formattedPercent = percentChange.toFixed(2);

                    return (
                      <tr
                        key={item.symbol || idx}
                        className={idx % 2 !== 0 ? 'bg-gray-900' : ''}
                        onClick={() => onRowClick(item)}
                        style={{ cursor: 'pointer' }}
                      >
                        <td className="py-2 px-4 text-sm">
                          <span className="font-medium text-gray-100">{item.symbol}</span>
                          {item.name && (
                            <span className="ml-2 text-gray-400 text-xs hidden sm:inline">
                              {item.name}
                            </span>
                          )}
                        </td>
                        <td className="py-2 px-4 text-sm text-right">{item.total_quantity}</td>
                        <td className="py-2 px-4 text-sm text-right">
                          ${item.average_price.toFixed(2)} → <b>${item.current_price.toFixed(2)}</b>
                        </td>
                        <td className="py-2 px-4 text-sm text-right">${value.toFixed(2)}</td>
                        <td className="py-2 px-4 text-sm text-right">
                        {parseFloat(percentChange.toFixed(2)) > 0 ? (
                          <span className="text-green-500 flex items-center justify-end">
                            <TrendingUp className="w-4 h-4 mr-1" />
                            +{formattedPercent}%
                          </span>
                        ) : parseFloat(percentChange.toFixed(2)) < 0 ? (
                          <span className="text-red-500 flex items-center justify-end">
                            <TrendingDown className="w-4 h-4 mr-1" />
                            {formattedPercent}%
                          </span>
                        ) : (
                          <span className="text-orange-500 flex items-center justify-end">
                            ={parseFloat(percentChange.toFixed(2))}%
                          </span>
                        )}
                        </td>
                      </tr>
                    );
                  })
                : (
                  <tr>
                    <td colSpan="5" className="py-4 px-4 text-center text-gray-500">
                      No data available.
                    </td>
                  </tr>
                )
            }
          </tbody>
        </table>
      </div>
    </div>
  );
}
