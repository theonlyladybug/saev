module Classification exposing (..)

import Array
import Browser
import Dict
import Gradio
import Html
import Html.Attributes
import Html.Events
import Http
import Json.Decode as D
import Json.Encode as E
import Set


main =
    Browser.document
        { init = init
        , view = view
        , update = update
        , subscriptions = \model -> Sub.none
        }



-- MESSAGE


type Msg
    = HoverPatch Int
    | ResetHoveredPatch
    | ToggleSelectedPatch Int
    | GotImageUrl (Result Gradio.Error (Maybe String))
    | GotPred (Result Gradio.Error (Dict.Dict String Float))
    | GotModified (Result Gradio.Error (Dict.Dict String Float))
    | GotSaeExamples (Result Gradio.Error (List SaeExampleResult))
    | SetSlider Int String



-- MODEL


type alias Model =
    { err : Maybe String
    , exampleIndex : Int
    , imageUrl : Maybe String
    , hoveredPatchIndex : Maybe Int
    , selectedPatchIndices : Set.Set Int
    , saeLatents : List SaeLatent
    , sliders : Dict.Dict Int Float

    -- ML stuff
    , preds : Dict.Dict String Float
    , modified : Dict.Dict String Float

    -- Progression
    , nPatchesExplored : Int
    , nPatchResets : Int
    , nImagesExplored : Int
    }


type alias SaeLatent =
    { latent : Int
    , urls : List String
    }


init : () -> ( Model, Cmd Msg )
init _ =
    let
        example =
            0
    in
    ( { err = Nothing
      , exampleIndex = example
      , imageUrl = Nothing
      , hoveredPatchIndex = Nothing
      , selectedPatchIndices = Set.empty
      , saeLatents = []
      , sliders = Dict.empty

      -- ML
      , preds = Dict.empty
      , modified = Dict.empty

      -- Progression
      , nPatchesExplored = 0
      , nPatchResets = 0
      , nImagesExplored = 0
      }
    , Cmd.batch
        [ getImageUrl example, getPred example ]
    )



-- UPDATE


update : Msg -> Model -> ( Model, Cmd Msg )
update msg model =
    case msg of
        HoverPatch i ->
            ( { model | hoveredPatchIndex = Just i }, Cmd.none )

        ResetHoveredPatch ->
            ( { model | hoveredPatchIndex = Nothing }, Cmd.none )

        GotImageUrl result ->
            case result of
                Ok url ->
                    ( { model | imageUrl = url }, Cmd.none )

                Err err ->
                    ( { model | imageUrl = Nothing, err = Just (explainGradioError err) }, Cmd.none )

        GotPred result ->
            case result of
                Ok preds ->
                    ( { model | preds = preds }, Cmd.none )

                Err err ->
                    ( { model | preds = Dict.empty, err = Just (explainGradioError err) }, Cmd.none )

        GotModified result ->
            case result of
                Ok preds ->
                    ( { model | modified = preds }, Cmd.none )

                Err err ->
                    ( { model | modified = Dict.empty, err = Just (explainGradioError err) }, Cmd.none )

        ToggleSelectedPatch i ->
            let
                ( patchIndices, nPatchesExplored, nPatchResets ) =
                    if Set.member i model.selectedPatchIndices then
                        ( Set.remove i model.selectedPatchIndices
                        , model.nPatchesExplored
                        , if Set.size model.selectedPatchIndices == 1 then
                            model.nPatchResets + 1

                          else
                            model.nPatchResets
                        )

                    else
                        ( Set.insert i model.selectedPatchIndices
                        , model.nPatchesExplored + 1
                        , model.nPatchResets
                        )
            in
            ( { model
                | selectedPatchIndices = patchIndices
                , nPatchesExplored = nPatchesExplored
                , nPatchResets = nPatchResets
              }
            , getSaeExamples model.exampleIndex patchIndices
            )

        GotSaeExamples result ->
            case result of
                Ok results ->
                    let
                        latents =
                            parseSaeExampleResults results

                        sliders =
                            latents
                                |> List.map (\latent -> ( latent.latent, 0.0 ))
                                |> Dict.fromList
                    in
                    ( { model
                        | saeLatents = latents
                        , sliders = sliders
                        , err = Nothing
                      }
                    , Cmd.none
                    )

                Err err ->
                    ( { model | err = Just (explainGradioError err) }, Cmd.none )

        SetSlider i str ->
            case String.toFloat str of
                Just f ->
                    let
                        sliders =
                            Dict.insert i f model.sliders
                    in
                    ( { model | sliders = sliders }
                    , getModifiedLabels model.exampleIndex sliders
                    )

                Nothing ->
                    ( model, Cmd.none )



-- API


explainGradioError : Gradio.Error -> String
explainGradioError err =
    case err of
        Gradio.NetworkError msg ->
            "Network error: " ++ msg

        Gradio.JsonError msg ->
            "Error decoding JSON: " ++ msg

        Gradio.ParsingError msg ->
            "Error parsing API response: " ++ msg

        Gradio.ApiError msg ->
            "Error in the API: " ++ msg


getImageUrl : Int -> Cmd Msg
getImageUrl img =
    Gradio.get "get-image"
        [ E.int img ]
        (D.list imgUrlDecoder |> D.map List.head)
        GotImageUrl


getPred : Int -> Cmd Msg
getPred img =
    Gradio.get "get-preds"
        [ E.int img ]
        (D.list labelsDecoder
            |> D.map List.head
            |> D.map (Maybe.withDefault Dict.empty)
        )
        GotPred


getModifiedLabels : Int -> Dict.Dict Int Float -> Cmd Msg
getModifiedLabels img sliders =
    let
        ( latents, values ) =
            sliders
                |> Dict.toList
                |> List.unzip
                |> Tuple.mapBoth Array.fromList Array.fromList

        args =
            [ E.int img
            , E.int (Array.get 0 latents |> Maybe.withDefault -1)
            , E.int (Array.get 1 latents |> Maybe.withDefault -1)
            , E.int (Array.get 2 latents |> Maybe.withDefault -1)
            , E.float (Array.get 0 values |> Maybe.withDefault 0.0)
            , E.float (Array.get 1 values |> Maybe.withDefault 0.0)
            , E.float (Array.get 2 values |> Maybe.withDefault 0.0)
            ]
    in
    Gradio.get "get-modified"
        [ E.int img ]
        (D.list labelsDecoder
            |> D.map List.head
            |> D.map (Maybe.withDefault Dict.empty)
        )
        GotModified


labelsDecoder : D.Decoder (Dict.Dict String Float)
labelsDecoder =
    D.field "confidences"
        (D.list
            (D.map2 Tuple.pair
                (D.field "label" D.string)
                (D.field "confidence" D.float)
            )
            |> D.map Dict.fromList
        )


getSaeExamples : Int -> Set.Set Int -> Cmd Msg
getSaeExamples img patches =
    Gradio.get
        "get-sae-examples"
        [ E.int img
        , Set.toList patches |> E.list E.int
        ]
        (D.list saeExampleResultDecoder)
        GotSaeExamples


type SaeExampleResult
    = SaeExampleMissing
    | SaeExampleUrl String
    | SaeExampleLatent Int


saeExampleResultDecoder : D.Decoder SaeExampleResult
saeExampleResultDecoder =
    D.oneOf
        [ imgUrlDecoder |> D.map SaeExampleUrl
        , D.int |> D.map SaeExampleLatent
        , D.null SaeExampleMissing
        ]


imgUrlDecoder : D.Decoder String
imgUrlDecoder =
    D.field "url" D.string
        |> D.map (String.replace "gradio_api/gradio_api" "gradio_api")
        |> D.map (String.replace "gra/gradio_api" "gradio_api")
        |> D.map (String.replace "gradio_ap/gradio_api" "gradio_api")
        |> D.map (String.replace "gradio_api/ca/gradio_api" "gradio_api")



-- VIEW


view : Model -> Browser.Document Msg
view model =
    { title = "Image Classification"
    , body =
        [ Html.header [] []
        , Html.main_
            []
            [ Html.div
                [ Html.Attributes.class "flex flex-row items-center" ]
                [ viewGriddedImage model.hoveredPatchIndex model.selectedPatchIndices model.imageUrl "Input Image"
                , viewPreds model.preds
                , viewPreds model.modified
                ]
            , Html.div
                [ Html.Attributes.style "display" "flex"
                , Html.Attributes.style "flex-direction" "row"
                ]
                [ viewSaeExamples model.selectedPatchIndices model.saeLatents model.sliders
                ]
            , viewErr model.err
            ]
        ]
    }


viewGriddedImage : Maybe Int -> Set.Set Int -> Maybe String -> String -> Html.Html Msg
viewGriddedImage hovered selected maybeUrl caption =
    case maybeUrl of
        Nothing ->
            Html.div [] [ Html.text "Loading..." ]

        Just url ->
            Html.div []
                [ Html.div
                    [ Html.Attributes.class "relative inline-block" ]
                    [ Html.div
                        [ Html.Attributes.class "absolute grid grid-rows-[repeat(14,_16px)] grid-cols-[repeat(14,_16px)] md:grid-rows-[repeat(14,_24px)] md:grid-cols-[repeat(14,_24px)]" ]
                        (List.map
                            (viewGridCell hovered selected)
                            (List.range 0 195)
                        )
                    , Html.img
                        [ Html.Attributes.class "block w-[224px] h-[224px] md:w-[336px] md:h-[336px]"
                        , Html.Attributes.src url
                        ]
                        []
                    ]
                , Html.p
                    []
                    [ Html.text caption ]
                ]


viewGridCell : Maybe Int -> Set.Set Int -> Int -> Html.Html Msg
viewGridCell hovered selected self =
    let
        classes =
            (case hovered of
                Just h ->
                    if h == self then
                        [ "border-2 border-rose-600 border-dashed" ]

                    else
                        []

                Nothing ->
                    []
            )
                ++ (if Set.member self selected then
                        [ "bg-rose-600/50" ]

                    else
                        []
                   )
    in
    Html.div
        ([ Html.Attributes.class "w-[16px] h-[16px] md:w-[24px] md:h-[24px]"
         , Html.Events.onMouseEnter (HoverPatch self)
         , Html.Events.onMouseLeave ResetHoveredPatch
         , Html.Events.onClick (ToggleSelectedPatch self)
         ]
            ++ List.map Html.Attributes.class classes
        )
        []


viewPreds : Dict.Dict String Float -> Html.Html Msg
viewPreds preds =
    let
        top5 =
            Dict.toList preds |> List.sortBy Tuple.second |> List.reverse |> List.take 5
    in
    Html.div
        []
        (List.map (uncurry viewPred) top5)


viewPred : String -> Float -> Html.Html Msg
viewPred label score =
    Html.div
        []
        [ Html.span [] [ Html.text label ]
        , Html.span [] [ Html.text (String.fromFloat score) ]
        ]


viewSaeExamples : Set.Set Int -> List SaeLatent -> Dict.Dict Int Float -> Html.Html Msg
viewSaeExamples selected latents values =
    if List.length latents > 0 then
        Html.div []
            ([ Html.p []
                [ Html.span [ Html.Attributes.class "bg-rose-600 p-1 rounded" ] [ Html.text "These patches" ]
                , Html.text " above are like "
                , Html.span [ Html.Attributes.class "bg-rose-600 text-white p-1 rounded" ] [ Html.text "these patches" ]
                , Html.text " below. (Not what you expected? Add more patches and get a larger "
                , Html.a [ Html.Attributes.href "https://simple.wikipedia.org/wiki/Sampling_(statistics)", Html.Attributes.class "text-blue-500 underline" ] [ Html.text "sample size" ]
                , Html.text ")"
                ]
             ]
                ++ List.filterMap
                    (\latent -> Maybe.map (\f -> viewSaeLatentExamples latent f) (Dict.get latent.latent values))
                    latents
            )

    else if Set.size selected > 0 then
        Html.p []
            [ Html.text "Loading similar patches..." ]

    else
        Html.p []
            [ Html.text "Click on the image above to explain model predictions." ]


viewSaeLatentExamples : SaeLatent -> Float -> Html.Html Msg
viewSaeLatentExamples latent value =
    Html.div
        [ Html.Attributes.class "flex flex-row gap-2 mt-2" ]
        (List.map viewImage latent.urls
            ++ [ Html.div
                    [ Html.Attributes.class "flex flex-col gap-2" ]
                    [ Html.input
                        [ Html.Attributes.type_ "range"
                        , Html.Attributes.min "-10"
                        , Html.Attributes.max "10"
                        , Html.Attributes.value (String.fromFloat value)
                        , Html.Events.onInput (SetSlider latent.latent)
                        ]
                        []
                    , Html.p
                        []
                        [ Html.text ("Latent 24K/" ++ String.fromInt latent.latent ++ ": " ++ String.fromFloat value) ]
                    ]
               ]
        )


viewImage : String -> Html.Html Msg
viewImage url =
    Html.img
        [ Html.Attributes.src url
        , Html.Attributes.class "max-w-36 h-auto"
        ]
        []


viewErr : Maybe String -> Html.Html Msg
viewErr err =
    case err of
        Just msg ->
            Html.p [] [ Html.text msg ]

        Nothing ->
            Html.span [] []



-- HELPERS


uncurry : (a -> b -> c) -> ( a, b ) -> c
uncurry f ( a, b ) =
    f a b


parseSaeExampleResults : List SaeExampleResult -> List SaeLatent
parseSaeExampleResults results =
    let
        ( _, _, parsed ) =
            parseSaeExampleResultsHelper results [] []
    in
    parsed


parseSaeExampleResultsHelper : List SaeExampleResult -> List (Maybe String) -> List SaeLatent -> ( List SaeExampleResult, List (Maybe String), List SaeLatent )
parseSaeExampleResultsHelper unparsed urls parsed =
    case unparsed of
        [] ->
            ( [], urls, parsed )

        (SaeExampleUrl url) :: rest ->
            parseSaeExampleResultsHelper
                rest
                (urls ++ [ Just url ])
                parsed

        (SaeExampleLatent latent) :: rest ->
            parseSaeExampleResultsHelper
                rest
                (List.drop nSaeExamplesPerLatent urls)
                (parsed
                    ++ [ { latent = latent
                         , urls =
                            urls
                                |> List.take nSaeExamplesPerLatent
                                |> List.filterMap identity
                         }
                       ]
                )

        SaeExampleMissing :: rest ->
            parseSaeExampleResultsHelper
                rest
                (urls ++ [ Nothing ])
                parsed



-- CONSTANTS


nSaeExamplesPerLatent =
    4
