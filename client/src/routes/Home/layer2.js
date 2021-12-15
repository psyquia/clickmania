import React, { useState, useEffect } from 'react';
import PropTypes from 'prop-types';
import ScrollMenu from 'react-horizontal-scrolling-menu';
import { useNavigate } from 'react-router-dom';
import client from 'api/client';
import layer1RightIm2 from 'assets/homeRtwo.png';
import { useUser } from '../../context/userContext';

import './layer2_scroll.css';
import './layer2.css';

const Layer2 = ({ data, collectionArray }) => {
	const navigate = useNavigate();
	//const ExploreUrl = user ? '/store' : '/signIn';
	const { user } = useUser();
	const [list, setList] = useState([]);
	const [colDict, setColDict] = useState({});

	const colors = [
		'#2a5caa',
		'#ef5b9c',
		'#8f4b2e',
		'#2e3a1f',
		'#d64f44',
		'#769149',
		'#585eaa',
		'#845538',
		'#2e3a1f',
		'#2b6447',
		'#65c294',
		'#b7704f',
		'#74787c',
	];

	// const buttonOnClick = (artID) => {
	// 	console.log(artID);
	// 	alert('WILL GO TO THE DETAILED PAGE! ART ID = '+ artID); // eslint-disable-line no-alert
	// };

	const randomColor = () => {
		const color = colors[Math.floor(Math.random() * colors.length)];
		return color;
	};

	const createFlexItem = () => {
		const list0 = [];
		if (data && data.length) {
			// console.log(data );
			data.map((art, index) =>
				list0.push({
					// eslint-disable-next-line
					index: index,
					imgUrl: art.img_url,
					backgroundColor: randomColor(),
					artTitle: art.title,
					artID: art.artID,
					rarity: art.rarity,
					collectionID: art.collection,
				})
			);

			// return list0;
		} else {
			for (let i = 0; i < 5; i += 1) {
				list0.push({
					index: i,
					imgUrl: layer1RightIm2,
					backgroundColor: randomColor(),
					artTitle: 'Anonymous',
					artID: -i,
					rarity: -1,
					collectionID: -1,
				});
			}
			// return list0;
		}

		setList(list0);
	};

	const createCollections = () => {
		const col0 = {};
		if (collectionArray && collectionArray.length) {
			collectionArray.map(
				// eslint-disable-next-line
				(collection) =>
					// (col0[collection.collectionID] = {
					// 	name: collection.display_name,
					// 	description: collection.description,
					// })
					(col0[collection.collectionID] = collection.display_name)
			);
		} else {
			col0[-1] = 'Anonymous';
			// col0[-1] = { name: 'Anonymous', description: 'Explore in the Art Store' };
		}
		// }

		setColDict(col0);
	};
	// MUST
	// eslint-disable-next-line
	const MenuItem = ({ art, col_name, img }) => {
		// const buttonOnClick = (artID) => {
		// 	// console.log(artID);
		// 	// alert('WILL GO TO THE DETAILED PAGE! ART ID = ' , artID); // eslint-disable-line no-alert
		// 	navigate(ExploreUrl);
		// };
		// <div className="scroll-flex-iback-flex" onClick={() => navigate(user ? '/store' : '/signIn')
		return (
			<div className="scroll-flex-iback-flex">
				<img src={img} alt="123" />
				<div className="scroll-flex-iback-footer-flex" style={{ backgroundColor: art.backgroundColor }}>
					<p className="scroll-flex-iback-footer-title">{art.artTitle} </p>
					<p className="scroll-flex-iback-footer-middle">{col_name} </p>
				</div>{' '}
			</div>
		);
	};

	// MUST
	// export const Menu = (list, selected) =>
	const Menu = (list_, colDict_) =>
		list_.map((art, index) => (
			// return (
			<MenuItem
				art={art}
				col_name={colDict_[art.collectionID]}
				img={art.collectionID < 0 ? layer1RightIm2 : client.getImg(art.imgUrl)}
				// eslint-disable-next-line
				key={index}
				// key={art.artTitle}
				// selected={selected}
				// onClick={onClick}
			/>
			// )
		));
	
	// eslint-disable-next-line	
	const onSelect = key => {
	 navigate(user ? '/store' : '/signIn');
   };

	//MUST
	// eslint-disable-next-line
	const Arrow = ({ text, className }) => {
		return <div className={className}> {text} </div>;
	};

	Arrow.propTypes = {
		// eslint-disable-next-line
		text: PropTypes.string,
		// eslint-disable-next-line
		className: PropTypes.string,
	};

	// MUST
	const ArrowLeft = Arrow({ text: '◀', className: 'arrow-prev' });
	const ArrowRight = Arrow({ text: '▶', className: 'arrow-next' });

	useEffect(() => {
		createFlexItem();
		createCollections();
	}, [data, collectionArray]);

	const menu = Menu(list, colDict);

	return (
		<div className="section2-flex">
			<div className="section-title-flex">
				<h2 className="section-title-flex-item">Notable Drops</h2>
			</div>
			<ScrollMenu
				alignCenter={false} // 这个aligncenter才是避免在每次窗口变动的时候，起始位置的offset一直在调整的主要原因。必须设置为false
				// dragging={true} 
				wheel={false}
				alignOnResize={false}
				hideArrows={false}
				data={menu}
				arrowLeft={ArrowLeft}
				arrowRight={ArrowRight}
				onSelect={onSelect}
			/>{' '}
		</div>
	);
};

export default Layer2;
