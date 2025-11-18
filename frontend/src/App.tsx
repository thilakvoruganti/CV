import { HashRouter, Route, Routes } from "react-router-dom";
import NavBar from "./components/NavBar";
import HomePage from "./pages/HomePage";
import ImageStitchingPage from "./pages/ImageStitchingPage";
import SiftComparePage from "./pages/SiftComparePage";
import ModuleTwoPage from "./pages/ModuleTwoPage";
import ModuleThreePage from "./pages/ModuleThreePage";
import MeasurePage from "./pages/MeasurePage";
import "./App.css";

function App() {
  return (
    <HashRouter>
      <div className="app-shell">
        <NavBar />
        <main className="app-content">
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/stitch" element={<ImageStitchingPage />} />
            <Route path="/sift" element={<SiftComparePage />} />
            <Route path="/module2" element={<ModuleTwoPage />} />
            <Route path="/module3" element={<ModuleThreePage />} />
            <Route path="/measure" element={<MeasurePage />} />
          </Routes>
        </main>
      </div>
    </HashRouter>
  );
}

export default App;
