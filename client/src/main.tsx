import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "./index.css";
import { Toaster } from "react-hot-toast";

// import { BrowserRouter, Route, Routes } from "react-router";

import App from "./App.tsx";
// import { Chat } from "./components/Chat/Chat.tsx";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <Toaster />
    <App />
  </StrictMode>
);
