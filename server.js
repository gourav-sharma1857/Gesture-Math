// server.js
import express from "express";
import fetch from "node-fetch";
import dotenv from "dotenv";
dotenv.config();

const app = express();
const PORT = 3000;

app.get("/api/data", async (req, res) => {
  const response = await fetch(`https://api.example.com?key=${process.env.API_KEY}`);
  const data = await response.json();
  res.json(data);
});

app.listen(PORT, () => console.log(`Server running on port ${PORT}`));
