import React, { useEffect, useState } from 'react';
import { Navbar, Card, Filterbar } from 'components';
import { css } from '@emotion/css';
import client from 'api/client';
import { useUser } from 'context/userContext';
import './MyCollection.css';
import ClipLoader from 'react-spinners/ClipLoader';

const MyCollection = () => {
	const sortByName = () => {
		setOwnedArt((v) =>
			// sorts by name, then collection, then rarity
			[...v].sort(
				(a, b) =>
					a.art.title.localeCompare(b.art.title) ||
					a.collection_name.localeCompare(b.collection_name) ||
					-a.art.rarity - -b.art.rarity
			)
		);
	};
	const sortByCollection = () => {
		setOwnedArt((v) =>
			// sorts by collection, then rarity, then name
			[...v].sort(
				(a, b) =>
					a.collection_name.localeCompare(b.collection_name) ||
					-a.art.rarity - -b.art.rarity ||
					a.art.title.localeCompare(b.art.title)
			)
		);
	};
	const sortByRarity = () => {
		setOwnedArt((v) =>
			// sorts by rarity, then collection, then name
			[...v].sort(
				(a, b) =>
					-a.art.rarity - -b.art.rarity ||
					a.collection_name.localeCompare(b.collection_name) ||
					a.art.title.localeCompare(b.art.title)
			)
		);
	};

	const handleFilterChange = (e) => {
		setFilterQuery(e.target.value);
		const q = e.target.value;
		setOwnedArt(() =>
			// filter by title and collection name
			allOwnedArt.filter(
				(a) =>
					a.art.title.toLowerCase().includes(q.toLowerCase()) ||
					a.collection_name.toLowerCase().includes(q.toLowerCase())
			)
		);
	};

	// Sort types: 0 = don't sort, 1 = by name, 2 = by collection, 3 = by rarity
	const sort = {
		0: () => 0,
		1: sortByName,
		2: sortByCollection,
		3: sortByRarity,
	};

	const [sortOrder, setSortOrder] = useState(0);
	const [artFetched, setArtFetched] = useState(false);
	const [allOwnedArt, setAllOwnedArt] = useState([]);
	const [ownedArt, setOwnedArt] = useState([]);
	const [filterQuery, setFilterQuery] = useState('');
	const [forceUpdate, setForceUpdate] = useState(0);
	const { user } = useUser();

	useEffect(async () => {
		const data = await client.post('auth/get_user_arts', { id: user.id });
		const ownedArtData = data?.data;
		setAllOwnedArt(ownedArtData);
		setOwnedArt(ownedArtData);
		setArtFetched(true);
		sort[sortOrder]();
	}, [forceUpdate]);

	useEffect(() => {
		sort[sortOrder]();
	}, [sortOrder]);

	useEffect(
		() => () => {
			setAllOwnedArt([]);
			setOwnedArt([]);
		},
		[]
	);

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
					<div className="header-right filter-container">
						<Filterbar handleFilterChange={handleFilterChange} filterQuery={filterQuery} />
					</div>

					<div className="header-left sort-left">
						<div className="sort-container">
							<h3 className="sort-txt">Sort by</h3>
							<div className="sort-btn-container">
								<button className="button-big sort-by-btn" onClick={() => setSortOrder(1)}>
									Name
								</button>
								<button className="button-big sort-by-btn" onClick={() => setSortOrder(2)}>
									Collection
								</button>
								<button className="button-big sort-by-btn" onClick={() => setSortOrder(3)}>
									Rarity
								</button>
							</div>
						</div>
					</div>
					<h1 className="header-center">My Collection</h1>
				</div>

				<div
					className={css`
						width: 79.5%;
						display: grid;
						grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
						justify-items: center;
						align-items: center;
						@media (max-width: 60em) {
							width: 100%;
						}
					`}
				>
					{ownedArt.map((owned_art) => {
						const { artID, rarity, img_url, title } = owned_art.art;
						const { isOnSale, collection_name } = owned_art;
						return (
							<Card
								key={`${artID}-${title}-${owned_art.own.ownID}`}
								rarity={rarity}
								img_url={img_url}
								title={title}
								isOnSale={isOnSale}
								setForceUpdate={() => setForceUpdate((v) => v + 1)}
								collection_name={collection_name}
								own={owned_art.own}
							/>
						);
					})}

					{!allOwnedArt.length && (
						<div className="collection-spinner-container">
							<ClipLoader color="#008fcc" loading={!artFetched} size={150} />
						</div>
					)}
				</div>
			</div>
		</>
	);
};

export default MyCollection;
