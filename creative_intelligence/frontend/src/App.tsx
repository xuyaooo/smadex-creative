import { Routes, Route } from "react-router-dom";
import Navbar from "./components/Navbar";
import Footer from "./components/Footer";
import HomePage from "./pages/HomePage";
import StatsPage from "./pages/StatsPage";
import ExplorerPage from "./pages/ExplorerPage";
import PredictPage from "./pages/PredictPage";

export default function App() {
  return (
    <div className="min-h-screen flex flex-col">
      <Navbar />
      <div className="flex-1">
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/stats" element={<StatsPage />} />
          <Route path="/explorer" element={<ExplorerPage />} />
          <Route path="/predict" element={<PredictPage />} />
        </Routes>
      </div>
      <Footer />
    </div>
  );
}
