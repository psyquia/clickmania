import React from 'react';
import { css } from '@emotion/css';
import client from 'api/client';
import './Listings.css';
import { toast } from 'react-toastify';

const Listings = ({ listings, user, setUser, refreshListings }) => {
	const handleStopSelling = async (saleID) => {
		try {
			const data = await client.post('auth/stop_selling', { saleID });
			if (data?.data?.success) {
				toast.success('Successfully remove art from marketplace!', { autoClose: 2000 });
				refreshListings();
			}
		} catch (error) {
			console.log('***error', error);
			toast.error('Oops, something went wrong. Please try again later.');
		}
	};

	const handleBuyFromMarket = async (saleID) => {
		try {
			const data = await client.post('auth/buy_from_market', { saleID, buyerID: user.id });
			if (data?.data?.success) {
				toast.success('Congratulations! You now own this art.', { autoClose: 2000 });
				setUser(data?.data?.user);
				refreshListings();
			}
		} catch (error) {
			console.log('***error', error);
			toast.error(error?.response?.data?.detail || 'Oops, something went wrong. Please try again later.');
		}
	};

	return (
		<div
			className={css`
				max-width: 80em;
			`}
		>
			<table>
				<thead>
					<tr>
						<th id="col-art">Art</th>
						<th id="col-title">Title (Collection)</th>
						<th id="col-price">Price</th>
						<th id="col-seller">Seller</th>
						<th id="col-button" className="hidden">
							.
						</th>
					</tr>
				</thead>

				<tbody>
					{listings.map((listing) => (
						<tr key={listing.sale.saleID} className="listing-item">
							<td id="col-art">
								<img
									className={`artwork-img rarity-${listing.art.rarity}`}
									src={client.getImg(listing.art.thumb_url)}
									alt={listing.art.filename}
								/>
							</td>
							<td id="col-title">
								<span className={`rarity-${listing.art.rarity}`}>{listing.art.title}</span>
								<br></br>({listing.collection.display_name})
							</td>
							<td id="col-price">{listing.sale.price}</td>
							<td id="col-seller">{listing.seller.username}</td>
							{user.id === listing?.seller?.id ? (
								<td id="col-button">
									<button
										className="button-small sell"
										onClick={() => handleStopSelling(listing?.sale?.saleID)}
									>
										Remove
									</button>
								</td>
							) : (
								<td id="col-button">
									<button
										className="button-small buy"
										onClick={() => handleBuyFromMarket(listing?.sale?.saleID)}
									>
										Buy
									</button>
								</td>
							)}
						</tr>
					))}
					{!listings.length && (
						<tr>
							<td colSpan="5">...come back later for more listings...</td>
						</tr>
					)}
				</tbody>
			</table>
		</div>
	);
};

export default Listings;
