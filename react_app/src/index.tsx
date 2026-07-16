import React from 'react';
import ReactDOM from 'react-dom/client';
import { BrowserRouter, Routes, Route } from 'react-router-dom'

// import pages
import Login from './pages/Login';
import Register from './pages/Register';
import Products from './pages/Products';
import Control from './pages/Control';
import Likes from './pages/Likes';
import ProductDetail from './pages/ProductDetail';
import Users from './pages/Users';
import Search from './pages/Search';


const routes = 
<BrowserRouter>
  <Routes>
    <Route path='/' element={<Login />} />
    <Route path='/register' element={<Register />} />
    <Route path='/products' element={<Control item={<Products />} />} />
    <Route path='/likes' element={<Control item={<Likes />} />} />
    <Route path='/productDetail/:id' element={<Control item={<ProductDetail />} />} />
    <Route path='/users' element={<Control item={<Users />} />} />
    <Route path='/search' element={<Control item={<Search />} />} />
  </Routes>
</BrowserRouter>

const root = ReactDOM.createRoot(
  document.getElementById('root') as HTMLElement
);
root.render(routes);

