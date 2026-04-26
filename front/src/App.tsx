import { useEffect } from "react";
import { Routes, Route, useLocation } from "react-router-dom";
import Navbar from "./components/Navbar";
import Footer from "./components/Footer";
import HomePage from "./pages/HomePage";
import StatsPage from "./pages/StatsPage";
import ExplorerPage from "./pages/ExplorerPage";
import PredictPage from "./pages/PredictPage";

function ScrollToTop() {
  const { pathname } = useLocation();
  useEffect(() => {
    window.scrollTo({ top: 0, left: 0, behavior: "auto" });
  }, [pathname]);
  return null;
}

export default function App() {
  return (
    <div className="min-h-screen flex flex-col">
      <ScrollToTop />
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
