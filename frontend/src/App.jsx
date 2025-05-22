import { Routes, Route, Navigate } from 'react-router-dom';
import Layout from './components/Layout';
import Portfolio from './pages/Portfolio';
import Transactions from './pages/Transactions';
import Prices from './pages/Prices';
import Risk from './pages/Risk';
import Ideas from './pages/Ideas';
import HistoricalChart from './pages/Chart';

function App() {
  return (
    <Routes>
      <Route path="/" element={<Layout />}>
        <Route index element={<Navigate to="/portfolio" replace />} />
        <Route path="portfolio" element={<Portfolio />} />
        <Route path="transactions" element={<Transactions />} />
        <Route path="chart" element={<HistoricalChart />} />
        <Route path="prices" element={<Prices />} />
        <Route path="risk" element={<Risk />} />
        <Route path="ideas" element={<Ideas />} />
      </Route>
    </Routes>
  );
}

export default App;
