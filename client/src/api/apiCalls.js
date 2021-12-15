import client from './client';

export const getAllCollections = async (noHeader = false) =>
	client
		.get('collection/', noHeader)
		.then((res) => res.data)
		.catch((err) => err);

export const getCollections = async (collectionID) =>
	client
		.get(`collection/${collectionID}/`)
		.then((res) => res.data)
		.catch((err) => err);

export const getAllArtwork = async (noHeader = false) =>
	client
		.get('art/', noHeader)
		.then((res) => res.data)
		.catch((err) => err);

export const getArtworkByCollection = async (collectionID) =>
	client
		.get('art/')
		.then((res) => res.data.filter((art) => art.collection === collectionID))
		.catch((err) => err);

export const getAllOwn = async () =>
	client
		.get('own/')
		.then((res) => res.data)
		.catch((err) => err);

export const getUserOwn = async (uID) =>
	client
		.get('own/')
		.then((res) => res.data.filter((own) => own.user === uID))
		.catch((err) => err);

export const getAllSales = async () =>
	client
		.get('sale/')
		.then((res) => res.data)
		.catch((err) => err);

export const getSale = async (saleID) =>
	client
		.get(`sale/${saleID}/`)
		.then((res) => res.data)
		.catch((err) => err);

export const getAllSaleListings = async () =>
	client
		.get('auth/get_listings')
		.then((res) => res.data)
		.catch((err) => err);

export const depositCurrency = async (user, currency) => {
	const tempUser = { ...user, coins: user.coins + currency };

	// api call to deposit coins
	try {
		const res = await client.post('auth/update_user', tempUser);
		return res.data;
	} catch (error) {
		return {};
	}
};

export const buyLootBox = async (usr, col) => {
	// api call to buy lootbox
	try {
		const res = await client.post('auth/buy_box', {
			user: usr,
			collection: col,
		});
		return res.data;
	} catch (error) {
		return { err: true };
	}
};
