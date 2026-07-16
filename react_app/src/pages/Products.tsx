import React, { useEffect, useState } from 'react'
import { allProduct } from '../services/productService'
import { iProduct } from '../models/iAllProduct'
import ProductItem from '../components/ProductItem'

function Products() {

  const [proArr, setproArr] = useState<iProduct[]>([])

  useEffect(() => {
    allProduct(1).then(res => {
      const dt = res.data
      const arr = dt.data
      setproArr(arr)
    })
  }, [])

  return (
    <>
      <h2>Products</h2>
      <div className='row'>
        {proArr.map((item, index) =>
          <div className='col-xs-12 col-sm-6 col-md-4 col-lg-3' key={index}>
            <ProductItem item={item}/>
          </div>
        )}
      </div>
    </>
  )
}

export default Products