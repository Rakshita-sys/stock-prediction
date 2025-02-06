"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts"

export default function Home() {
  const [ticker, setTicker] = useState("AAPL")
  const [startDate, setStartDate] = useState("2022-01-01")
  const [endDate, setEndDate] = useState("2023-01-01")
  const [data, setData] = useState(null)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    const response = await fetch("http://localhost:5000/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ ticker, start_date: startDate, end_date: endDate }),
    })
    const result = await response.json()
    setData(result)
  }

  return (
    <div className="container mx-auto p-4">
      <h1 className="text-3xl font-bold mb-4">Stock Price Prediction</h1>
      <form onSubmit={handleSubmit} className="mb-4">
        <div className="flex space-x-2">
          <Input type="text" value={ticker} onChange={(e) => setTicker(e.target.value)} placeholder="Stock Ticker" />
          <Input type="date" value={startDate} onChange={(e) => setStartDate(e.target.value)} />
          <Input type="date" value={endDate} onChange={(e) => setEndDate(e.target.value)} />
          <Button type="submit">Predict</Button>
        </div>
      </form>
      {data && (
        <Card>
          <CardHeader>
            <CardTitle>{ticker} Stock Price Prediction</CardTitle>
            <CardDescription>Comparing actual prices with Linear Regression and LSTM predictions</CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={400}>
              <LineChart
                data={data.dates.map((date, i) => ({
                  date,
                  actual: data.actual[i],
                  lr: data.lr_predictions[i],
                  lstm: data.lstm_predictions[i],
                }))}
                margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="actual" stroke="#8884d8" name="Actual" />
                <Line type="monotone" dataKey="lr" stroke="#82ca9d" name="Linear Regression" />
                <Line type="monotone" dataKey="lstm" stroke="#ffc658" name="LSTM" />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      )}
    </div>
  )
}

