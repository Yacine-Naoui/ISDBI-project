import { createRoot } from "react-dom/client";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import './index.css'
import LandingPage from "./pages/home/LandingPage";
import ReportPage from "./pages/report/ReportPage";
import StandardsPage from "./pages/standards/StandardsPage";

function App() {

  return (
    <>
      <BrowserRouter>
        <Routes>
          {/* Route for the prompt or chat page */}
          <Route exact path={"/prompt"} element={<LandingPage />} />

          {/* Route for the report page, with dynamic reportId */}
          <Route exact path={"/report/:reportId"} element={<ReportPage />} />
          <Route exact path={"/report"} element={<ReportPage />} />

          {/* Route for standards */}
          <Route exact path={"/stand"} element={<StandardsPage />} />
        </Routes>
      </BrowserRouter>
    </>
  )
}

export default App

const root = createRoot(document.getElementById("root"));
root.render(<App />);