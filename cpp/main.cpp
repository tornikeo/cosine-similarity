#include <istream>
#include <fstream>
#include <iostream>
#include <iterator>
#include <utility>
#include <unordered_set>

#include <xtensor/xarray.hpp>
#include <xtensor/xcsv.hpp>
#include <xtensor/xnpy.hpp>
#include <xtensor/xslice.hpp>
#include <xtensor/xview.hpp>

using namespace std;
using namespace xt::placeholders;

template<class Stream, class Vector, class Begin = decltype(std::begin(std::declval<Vector>()))>
inline Stream& operator<<(Stream& stream, const Vector& vect) {
	const char* dlm = "";
	for(const auto& i : vect) { stream << dlm << i; dlm = ", "; }
	return stream;
}

template <class E, class E2> 
inline auto trim_values(E&& e, const std::string& direction, E2&& e2)
{
    XTENSOR_ASSERT_MSG(e.dimension() == 1, "Dimension for trim_zeros has to be 1.");

    std::ptrdiff_t begin = 0, end = static_cast<std::ptrdiff_t>(e.size());

    auto find_fun = [e2](const auto& i) {
        return i != e2;
    };

    if (direction.find("f") != std::string::npos)
    {
        begin = std::find_if(e.cbegin(), e.cend(), find_fun) - e.cbegin();
    }

    if (direction.find("b") != std::string::npos && begin != end)
    {
        end -= std::find_if(e.crbegin(), e.crend(), find_fun) - e.crbegin();
    }

    return strided_view(std::forward<E>(e), { xt::range(begin, end) });
}

template<typename T>
const xt::xarray<float> load(string filename) {
    cout << "Opening " << filename << " file " << endl;
    std::ifstream in_file;
    in_file.open(filename);
    cout << "Loading file in memory" << endl;
    auto references = xt::load_npy<T>(in_file);
    cout << "Data file loaded. It has " << references.shape() << " shape" << endl;
    in_file.close();
    return references;
}

int main()
{   
    const auto output_path = "data/results.npy";
    const auto all_rmzs = load<float>("data/references_mz.npy");
    const auto all_rints = load<float>("data/references_int.npy");
    // const auto all_rints_bs = load<size_t>("references_batch_size.npy");


    const auto all_qmzs = load<float>("data/queries_mz.npy");
    const auto all_qints = load<float>("data/queries_int.npy");
    // const auto all_qints_bs = load<size_t>("queries_batch_size.npy");
    
    const auto fill_value = static_cast<float>(-1e6);

    const auto R = all_rmzs.shape()[0];
    const auto Q = all_qmzs.shape()[0];

    
    const auto tolerance = 0.1f;
    const auto shift = 0.0f;
    const auto mz_power = 0.0f;
    const auto intensity_power = 1.0f;
    // const auto T = static_cast<int>( max(M, N) );

    auto results = vector<double>();

    // cout << vector<unsigned long int>({R, M, Q, N}) << endl;
    for (size_t r = 0; r < R; r++)
    {
        auto rmzs = trim_values(xt::row(all_rmzs, r), "b", fill_value);
        auto rints = trim_values(xt::row(all_rints, r), "b", fill_value);
        // auto bs_size = xt::view(all_rints_bs, r);
        // auto rmzs = xt::view(all_rmzs, r, xt::range(_, bs_size));
        // auto rints = xt::view(all_rints, r, xt::range(_, bs_size));

        for (size_t q = 0; q < Q; q++)
        {

            // auto bs_size = xt::view(all_qints_bs, q);
            // auto rmzs = xt::view(all_rmzs, r, xt::range(_, bs_size));
            // auto rints = xt::view(all_rints, r, xt::range(_, bs_size));
            auto qmzs = trim_values(xt::row(all_qmzs, q), "b", fill_value);
            auto qints = trim_values(xt::row(all_qints, q), "b", fill_value);
            
            size_t lowest_idx = 0;

            // find_matches
            auto matches = vector< pair<size_t, size_t> >();
            for (size_t peak1_idx = 0; peak1_idx < rmzs.size(); peak1_idx++)
            {
                auto mz = rmzs(peak1_idx);
                // cout << xt::isin() << endl;
                assert(mz > fill_value);
                // if (mz <= fill_value) break;
                auto low_bound = mz - tolerance;
                auto high_bound = mz + tolerance;
                for (size_t peak2_idx = lowest_idx; peak2_idx < qmzs.size(); peak2_idx++)
                {
                    auto mz2 = qmzs(peak2_idx) + shift;
                    assert(mz2 > fill_value);

                    if (mz2 > high_bound) {
                        break;
                    }
                    if (mz2 < low_bound) {
                        lowest_idx = peak2_idx;
                    } else {
                        matches.push_back(pair(peak1_idx, peak2_idx));
                    }
                }
            }


            if (!matches.empty()) {

                // cout << "(" << 
                //     matches[0].first << "," << 
                //     matches[0].second << " ), (" <<
                //     matches[1].first << "," << 
                //     matches[1].second << ")," << 
                //     endl;
                
                auto idx1 = vector<size_t>();
                auto idx2 = vector<size_t>();

                for (pair i : matches)
                {
                    idx1.push_back(i.first);
                    idx2.push_back(i.second);
                }
                
                auto matching_pairs = vector< tuple<size_t, size_t, float> >();
                
                for (size_t i = 0; i < idx1.size(); i++)
                {
                    auto idx = matches[i].first;
                    auto idx2 = matches[i].second;
                    
                    auto spec1_mz = rmzs(idx);
                    auto spec1_int = rints(idx);

                    auto spec2_mz = qmzs(idx2);
                    auto spec2_int = qints(idx2);

                    auto power_prod_spec1 = pow(spec1_mz, mz_power) * pow(spec1_int, intensity_power);
                    auto power_prod_spec2 = pow(spec2_mz, mz_power) * pow(spec2_int, intensity_power);
                    // cout << "(" << idx << "," << idx2 << ","
                    //  << power_prod_spec1 << "," << power_prod_spec2 << ")" << endl;
                    // return 0;
                    const auto tmp = make_tuple(idx, idx2, power_prod_spec1 * power_prod_spec2);
                    matching_pairs.push_back(tmp);
                }


                // cout << "(" << 
                //     get<0>(matching_pairs[0]) << "," << 
                //     get<1>(matching_pairs[0]) << "," << 
                //     get<2>(matching_pairs[0]) << "),(" << 
                //     get<0>(matching_pairs[1]) << "," << 
                //     get<1>(matching_pairs[1]) << "," << 
                //     get<2>(matching_pairs[1]) << ")," << 
                //     endl;
                // return 0;

                auto score = 0.0f;
                auto used_matches = 0ul;
                auto used1 = std::unordered_set<size_t>();
                auto used2 = std::unordered_set<size_t>();

                for (size_t i = 0; i < matching_pairs.size(); i++)
                {
                    auto [idx, idx2, matching_score] = matching_pairs[i];
                    if ( used1.find(idx) == used1.end() && used2.find(idx2) == used2.end() ) {
                            score += matching_score;
                            used1.insert(idx);
                            used2.insert(idx2);
                            used_matches++;
                    }
                }
                // cout << score << endl;
                // return 0;
                // cout << score << endl;
                auto spec1_power = xt::pow(rmzs, mz_power) * xt::pow(rints, intensity_power);
                auto spec2_power = xt::pow(qmzs, mz_power) * xt::pow(qints, intensity_power);
                // cout << "Let me guess, the error is here..." << endl;
                // TODO: Inefficient. Two calls to sum()...
                // auto x = xt::sum(xt::square(spec1_power), 0);
                // cout << "NOPE1" << endl;
                // auto z = xt::square(spec2_power);
                // cout << "NOPE2" << endl;
                // auto y = xt::sum(z, 0);
                // cout << "NOPE3" << endl;
                // auto score_norm = sqrt(x * y);

                auto score_norm = sqrt(xt::sum(xt::square(spec1_power)) * xt::sum(xt::square(spec2_power)));
                // cout << std::vector<double>(spec1_power.begin(), spec1_power.end()) << endl;
                
                // cout << std::vector<double>(spec2_power.begin(), spec2_power.end()) << endl;

                // cout << score << "/" << score_norm << endl;
                // return 0;
                
                // cout << "(" << spec1_power[0] << ","  
                //             << spec1_power[1] << "," 
                //             << spec1_power[2] << "," 
                //             << spec1_power[3] << "," 
                //             << spec1_power[4] << "...),(" 
                //             << spec2_power[0] << "," 
                //             << spec2_power[1] << "," 
                //             << spec2_power[2] << "," 
                //             << spec2_power[3] << "," 
                //             << spec2_power[4] << "...)," << 
                //     endl;
                // return 0;
                // auto score_norm = xt::sum(spec1_power) * xt::sum(spec2_power);
                // auto score_norm = 0.0f;
                // for(auto it=spec1_power.begin(); it!=spec1_power.end(); ++it)
                // {
                //     score_norm += *it;
                // }
                
                // auto score_norm = xt::sum(spec1_power) * xt::sum(spec2_power);
                auto score_norm_result = score_norm();
                score = score / score_norm_result;
                results.insert(results.end(), {score, static_cast<double>(used_matches)});
            }
        }
    }
    
    // auto [Q, N] = qmz.shape().data();

    // cout << "Opening queries.npy file " << endl;
    // in_file.open("queries.npy");
    // cout << "Loading file in memory" << endl;
    // auto queries = xt::load_npy<float>(in_file);
    // cout << "Data file loaded. It has " << queries.shape() << " shape" << endl;
    // in_file.close();

    // auto ref_mz = xt::xview(references, xt::all(), xt::all(), 0);

    // for (size_t ref_id = 0; ref_id < references.shape()[0]; ref_id++)
    // {
    //     for (size_t q_id = 0; q_id < queries.shape()[0]; q_id++)
    //     {
    //         auto ref = references(ref_id);
    //         /* code */
    //     }
    // }

    cout << "Dumping results.npy..." << endl;
    auto xresults = xt::adapt(results, {results.size()/2, 2ul});
    xt::dump_npy(output_path, xresults);
    
    cout << "Done!" << endl;
    // xt::xarray<float> a = {{1,2,3,4}, {5,6,7,8}};
    // xt::dump_npy("out.npy", a);

    return 0;
}
