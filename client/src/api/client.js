import axios from 'axios';

const baseUrl = process.env.REACT_APP_API_URL || 'http://localhost/api/';

const options = {
	headers: {
		Authorization: `Token ${localStorage.getItem('access-token')}`,
		'Content-Type': 'application/json',
	},
};

const client = {
	get: (endpoint, noHeader = false) =>
		noHeader ? axios.get(baseUrl + endpoint) : axios.get(baseUrl + endpoint, options),
	getImg: (endpoint) => baseUrl.slice(0, -5) + endpoint,
	post: (endpoint, data, noHeader = false) =>
		noHeader ? axios.post(baseUrl + endpoint, data) : axios.post(baseUrl + endpoint, data, options),
	remove: (endpoint) => axios.delete(baseUrl + endpoint, options),
	updateToken: () => {
		options.headers.Authorization = `Token ${localStorage.getItem('access-token')}`;
	},
};

export default client;
