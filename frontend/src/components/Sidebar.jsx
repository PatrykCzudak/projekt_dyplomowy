import { NavLink } from 'react-router-dom';
import { PieChart, List, DollarSign, AlertTriangle, Lightbulb } from 'lucide-react';

export default function Sidebar() {
  const baseClasses = "flex items-center px-3 py-2 rounded-md transition-colors duration-200";
  const activeClasses = "bg-gray-800 text-white font-medium";
  const inactiveClasses = "text-gray-400 hover:bg-gray-800 hover:text-white";

  return (
    <nav className="w-64 bg-gray-950 p-4">
      <div className="text-xl font-bold text-gray-100 pb-3">My Dashboards</div>
      <ul className="space-y-1">
        <li>
          <NavLink to="/portfolio" className={({ isActive }) => 
            `${baseClasses} ${isActive ? activeClasses : inactiveClasses}`
          }>
            <PieChart className="w-4 h-4 mr-2" /> Portfolio Management
          </NavLink>
        </li>
        <li>
          <NavLink to="/transactions" className={({ isActive }) => 
            `${baseClasses} ${isActive ? activeClasses : inactiveClasses}`
          }>
            <List className="w-4 h-4 mr-2" /> Transaction History
          </NavLink>
        </li>
        <li>
          <NavLink to="/chart" className={({ isActive }) => 
            `${baseClasses} ${isActive ? activeClasses : inactiveClasses}`
          }>
            <PieChart className="w-4 h-4 mr-2" /> Charts
          </NavLink>
        </li>
        <li>
          <NavLink to="/prices" className={({ isActive }) => 
            `${baseClasses} ${isActive ? activeClasses : inactiveClasses}`
          }>
            <DollarSign className="w-4 h-4 mr-2" /> Prices
          </NavLink>
        </li>
        <li>
          <NavLink to="/risk" className={({ isActive }) => 
            `${baseClasses} ${isActive ? activeClasses : inactiveClasses}`
          }>
            <AlertTriangle className="w-4 h-4 mr-2" /> Risk
          </NavLink>
        </li>
        <li>
          <NavLink to="/optimize" className={({ isActive }) => 
            `${baseClasses} ${isActive ? activeClasses : inactiveClasses}`
          }>
            <Lightbulb className="w-4 h-4 mr-2" /> Portfolio Optimization 
          </NavLink>
        </li>
      </ul>
    </nav>
  );
}
