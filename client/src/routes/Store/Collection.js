import React, { useState, useEffect } from 'react';
import { getArtworkByCollection, buyLootBox } from 'api/apiCalls';
import RotateLoader from 'react-spinners/RotateLoader';
import client from 'api/client';
import im from 'assets/open-case.png';
import { toast } from 'react-toastify';
import { useUser } from '../../context/userContext';
import './Collection.css';

const Collection = ({ collection }) => {
	const [artworks, setArtworks] = useState([]);
	const [buyAnimActive, setBuyAnimActive] = useState(false);
	const [buySpinner, setBuySpinner] = useState(false);
	const [viewCollection, setViewCollection] = useState(false);
	const [boxItem, setBoxItem] = useState({});
	const [showHint, setShowHint] = useState(false);
	const { user, setUser } = useUser();
	useEffect(() => {
		handleGetArtByCollection();
	}, []);

	const handleGetArtByCollection = async () => {
		const art = await getArtworkByCollection(collection.collectionID);
		if (!art.length) return;
		art.sort((a, b) => b.rarity - a.rarity);
		setArtworks(art);
	};

	const handleBuy = async () => {
		setBoxItem({});
		setBuyAnimActive(true);
		setBuySpinner(true);
		new Promise((r) => setTimeout(r, 1000)).then(() => setBuySpinner(false));
		const res = await buyLootBox(user, collection);

		if (res.err) {
			setBuySpinner(false);
			setBuyAnimActive(false);
			toast.error('An unexpected error occured.');
			return;
		}

		// insufficient funds
		if (!res.user) {
			setBuySpinner(false);
			setBuyAnimActive(false);
			toast.error(`${res} (${user.coins} coins)`);
			return;
		}
		setUser(res?.user);
		setBoxItem(res?.art);
	};

	const replaceNewline = (str) => str.replace('\\n', '\n');

	return (
		<div className="collection-container">
			<div className="box_img-container">
				<img className="box_img" src={client.getImg(collection.img_url)} alt="art" />
			</div>
			<div className="collection-info">
				<h3>
					{collection.display_name} Collection{' '}
					<span
						className="hover-hint"
						onMouseOver={() => setShowHint(true)}
						onMouseLeave={() => setShowHint(false)}
						onFocus={() => setShowHint(true)}
						onBlur={() => setShowHint(false)}
					>
						?
					</span>
				</h3>
				{showHint && (
					<div className="pack-probabilities-container" style={{ pointerEvents: 'none' }}>
						Box Probabilities:
						<p className="rarity-info">
							<em className="rarity-5">Legendary:</em> 1%
						</p>
						<p className="rarity-info">
							<em className="rarity-4">Very Rare:</em> 10%
						</p>
						<p className="rarity-info">
							<em className="rarity-3">Rare:</em> 25%
						</p>
						<p className="rarity-info">
							<em className="rarity-2">Uncommon:</em> 31%
						</p>
						<p className="rarity-info">
							<em className="rarity-1">Common:</em> 33%
						</p>
					</div>
				)}
				<div className="description-container">
					{replaceNewline(collection.description)}
					<div className="price-container">
						<p className="price-txt">Price:</p>
						<em className="price-val">100 ⓒ</em>
					</div>
				</div>
				<div className="button-container">
					<button className="button-big buy" onClick={handleBuy}>
						Buy One
					</button>
					<button className="button-big view" onClick={() => setViewCollection((v) => !v)}>
						View Collection
					</button>
				</div>
			</div>
			<div className={`popup-container ${viewCollection ? 'show' : 'hide'}`}>
				<div
					className="click-exit-layer"
					onClick={() => {
						if (!buySpinner) setViewCollection(false);
					}}
				/>
				<div className="collection-popup">
					{viewCollection && (
						<button className="close-btn-col" onClick={() => setViewCollection(false)}>
							x
						</button>
					)}
					{artworks.map((art) => (
						<div className="artwork-single" key={art.artID}>
							<img
								className={`artwork-img rarity-${art.rarity}`}
								src={client.getImg(art.thumb_url)}
								alt="art"
							/>
							<br />
							{art.title}
						</div>
					))}
				</div>
			</div>
			<div className={`popup-container ${buyAnimActive ? 'show' : 'hide'}`}>
				<div
					className="click-exit-layer"
					onClick={() => {
						if (!buySpinner) setBuyAnimActive(false);
					}}
				/>
				{buySpinner && (
					<div className="spinner-container">
						<div className="spinner-sdiv">
							<RotateLoader
								className="spinner"
								color="#00d15b"
								loading={buySpinner}
								size={28}
								margin={50}
								speedMultiplier={1}
							/>
						</div>
						<img className="open-case-img" src={im} alt="open case" />
					</div>
				)}
				<div className={`buy-popup ${buySpinner ? 'hide' : ''}`}>
					{boxItem && (
						<div className="box-item-container">
							<h1 className="you-got-art">You Got Art!</h1>
							<div className="box-item-img-container">
								<img
									className={`box-item-img rarity-${boxItem.rarity}`}
									src={client.getImg(boxItem.img_url)}
									alt="box-art"
								/>
							</div>
							<div className="box-item-detail-container">
								<h2 className="box-item-title">"{boxItem.title}"</h2>
								<h4 className="box-item-collection">{collection.display_name} Collection</h4>
							</div>
						</div>
					)}
					{buyAnimActive && (
						<button
							className="close-btn-buy"
							onClick={() => {
								if (!buySpinner) setBuyAnimActive(false);
							}}
						>
							x
						</button>
					)}
				</div>
			</div>
		</div>
	);
};
export default Collection;
