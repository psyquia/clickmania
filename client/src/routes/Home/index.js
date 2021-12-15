import React, { useState, useEffect } from 'react';
import { Navbar } from 'components';
import { getAllCollections, getAllArtwork } from 'api/apiCalls';
import { css } from '@emotion/css';
import Layer1 from './layer1';
import Layer2 from './layer2';

const Home = () => {
	const [collections, setCollections] = useState([]);
	const [artworks, setArtworks] = useState([]);
	const [isMounted1, setIsMounted1] = useState([]);
	const [isMounted2, setIsMounted2] = useState([]);

	// setIsMounted(true);

	useEffect(() => {
		setIsMounted1(true);
		const handleGetCollections = async () => {
			const col = await getAllCollections(true);
			if (!col.length) return;
			// reverse order
			col.sort((a, b) => b.collectionID - a.collectionID);
			if (isMounted1) {
				setCollections(col);
			}
		};
		handleGetCollections();
		return function cleanup() {
			setIsMounted1(false);
		};
	}, []);

	useEffect(() => {
		setIsMounted2(true);

		const handleGetAllArts = async () => {
			const art = await getAllArtwork(true);
			if (!art.length) return;
			art.sort((a, b) => b.rarity - a.rarity);
			if (isMounted2) {
				setArtworks(art);
			}
		};
		handleGetAllArts();

		return function cleanup() {
			setIsMounted2(false);
		};
	}, []);

	// useEffect(() => {
	// 	return () => {
	// 		setIsMounted(false);
	// 	}
	// }, []);

	// useEffect(() => {
	//     window.location.reload();
	// }, []);

	// const handleGetCollections = async() => {
	//     const col = await getAllCollections(true);
	//     if (!col.length) return;
	//     // reverse order
	//     col.sort((a, b) => b.collectionID - a.collectionID);
	//     setCollections(col);
	//     // initial(col.length);
	// };

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
				`}
			>
				<Layer1 data={collections} /> <Layer2 data={artworks} collectionArray={collections} />{' '}
			</div>{' '}
		</>
	);
};

export default Home;
