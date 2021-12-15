import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { Navbar, Filterbar } from 'components';
import { css } from '@emotion/css';
import { getAllSaleListings } from 'api/apiCalls';
import { useUser } from 'context/userContext';
import Listings from './Listings';
import './Trade.css';

const Trade = () => {
	const [allSaleListings, setAllSaleListings] = useState([]);
	const [saleListings, setSaleListings] = useState([]);
	const [filterQuery, setFilterQuery] = useState('');
	const [refresh, setRefresh] = useState(0);
	const { user, setUser } = useUser();

	const handleGetAllSaleListings = async () => {
		const saleListingsAll = await getAllSaleListings();
		if (!saleListingsAll.length) return;

		setAllSaleListings(saleListingsAll);
		setSaleListings(saleListingsAll);
	};

	const handleFilterChange = (e) => {
		setFilterQuery(e.target.value);
		const q = e.target.value;
		setSaleListings(() =>
			allSaleListings.filter(
				(sale) =>
					sale.art.title.toLowerCase().includes(q.toLowerCase()) ||
					sale.seller.username.toLowerCase().includes(q.toLowerCase()) ||
					sale.collection.display_name.toLowerCase().includes(q.toLowerCase())
			)
		);
	};

	useEffect(() => {
		handleGetAllSaleListings();
	}, [refresh]);

	return (
		<>
			<Navbar />
			<div
				className={css`
					width: 100%;
					display: flex;
					flex-direction: column;
					justify-content: center;
					align-items: center;
					text-align: center;
					margin-top: 30px;
				`}
			>
				<div className="header-container">
					<h1 className="header-center">Trade</h1>
					<div className="filterbar-container header-right">
						<Filterbar handleFilterChange={handleFilterChange} filterQuery={filterQuery} />
					</div>
				</div>
				<Listings
					listings={saleListings}
					user={user}
					setUser={setUser}
					refreshListings={() => setRefresh((v) => v + 1)}
				/>
				<Link to="/">Back to homepage</Link>
			</div>
		</>
	);
};

export default Trade;
