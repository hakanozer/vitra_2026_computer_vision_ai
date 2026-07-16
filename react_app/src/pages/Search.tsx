import React, { useEffect, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { searchProduct } from '../services/productService'
import { iProduct } from '../models/iAllProduct'
import ProductItem from '../components/ProductItem'

function Search() {

  const [proArr, setproArr] = useState<iProduct[]>([])
  const [params, setParams] = useSearchParams()
  useEffect(() => {
    const q = params.get('q')
    if (q) {
        searchProduct(q).then(res => {
            const dt = res.data
            setproArr(dt.data)
        })
    }
  }, [])
  

  return (
    <>
      <h2>Search</h2>
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

export default Search